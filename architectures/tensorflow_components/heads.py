#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf
import numpy as np
from utils import force_list


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class Head(object):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        self.head_idx = head_idx
        self.name = "head"
        self.output = []
        self.loss = []
        self.loss_type = []
        self.regularizations = []
        self.loss_weight = force_list(loss_weight)
        self.target = []
        self.input = []
        self.is_local = is_local

    def __call__(self, input_layer):
        """
        Wrapper for building the module graph including scoping and loss creation
        :param input_layer: the input to the graph
        :return: the output of the last layer and the target placeholder
        """
        with tf.variable_scope(self.get_name(), initializer=tf.contrib.layers.xavier_initializer()):
            self._build_module(input_layer)

            self.output = force_list(self.output)
            self.target = force_list(self.target)
            self.input = force_list(self.input)
            self.loss_type = force_list(self.loss_type)
            self.loss = force_list(self.loss)
            self.regularizations = force_list(self.regularizations)
            if self.is_local:
               self.set_loss()

        if self.is_local:
            return self.output, self.target, self.input
        else:
            return self.output, self.input

    def _build_module(self, input_layer):
        """
        Builds the graph of the module
        :param input_layer: the input to the graph
        :return: None
        """
        pass

    def get_name(self):
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return '{}_{}'.format(self.name, self.head_idx)

    def set_loss(self):
        """
        Creates a target placeholder and loss function for each loss_type and regularization
        :param loss_type: a tensorflow loss function
        :param scope: the name scope to include the tensors in
        :return: None
        """
        # add losses and target placeholder
        for idx in range(len(self.loss_type)):
            target = tf.placeholder('float', self.output[idx].shape, '{}_target'.format(self.get_name()))
            self.target.append(target)
            loss = self.loss_type[idx](self.target[-1], self.output[idx],
                                       weights=self.loss_weight[idx], scope=self.get_name())
            self.loss.append(loss)

        # add regularizations
        for regularization in self.regularizations:
            self.loss.append(regularization)


class QHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'q_values_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # Standard Q Network
        self.output = tf.layers.dense(input_layer, self.num_actions, name='output')


class DuelingQHead(QHead):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        QHead.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)

    def _build_module(self, input_layer):
        # state value tower - V
        with tf.variable_scope("state_value"):
            state_value = tf.layers.dense(input_layer, 256, activation=tf.nn.relu)
            state_value = tf.layers.dense(state_value, 1)
            # state_value = tf.expand_dims(state_value, axis=-1)

        # action advantage tower - A
        with tf.variable_scope("action_advantage"):
            action_advantage = tf.layers.dense(input_layer, 256, activation=tf.nn.relu)
            action_advantage = tf.layers.dense(action_advantage, self.num_actions)
            action_advantage = action_advantage - tf.reduce_mean(action_advantage)

        # merge to state-action value function Q
        self.output = tf.add(state_value, action_advantage, name='output')


class VHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'v_values_head'
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # Standard V Network
        self.output = tf.layers.dense(input_layer, 1, name='output',
                                            kernel_initializer=normalized_columns_initializer(1.0))


class PolicyHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'policy_values_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.output_scale = np.max(tuning_parameters.env_instance.action_space_abs_range)
        self.discrete_controls = tuning_parameters.env_instance.discrete_controls
        self.exploration_policy = tuning_parameters.exploration.policy
        self.exploration_variance = 2*self.output_scale*tuning_parameters.exploration.initial_noise_variance_percentage
        if not self.discrete_controls and not self.output_scale:
            raise ValueError("For continuous controls, an output scale for the network must be specified")
        self.beta = tuning_parameters.agent.beta_entropy

    def _build_module(self, input_layer):
        eps = 1e-15
        if self.discrete_controls:
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
        else:
            self.actions = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
        self.input = [self.actions]

        # Policy Head
        if self.discrete_controls:
            policy_values = tf.layers.dense(input_layer, self.num_actions)
            self.policy_mean = tf.nn.softmax(policy_values, name="policy")

            # define the distributions for the policy and the old policy
            # (the + eps is to prevent probability 0 which will cause the log later on to be -inf)
            self.policy_distribution = tf.contrib.distributions.Categorical(probs=(self.policy_mean + eps))
            self.output = self.policy_mean
        else:
            # mean
            policy_values_mean = tf.layers.dense(input_layer, self.num_actions, activation=tf.nn.tanh)
            self.policy_mean = tf.multiply(policy_values_mean, self.output_scale, name='output_mean')

            self.output = [self.policy_mean]

            # std
            if self.exploration_policy == 'ContinuousEntropy':
                policy_values_std = tf.layers.dense(input_layer, self.num_actions,
                                            kernel_initializer=normalized_columns_initializer(0.01))
                self.policy_std = tf.nn.softplus(policy_values_std, name='output_variance') + eps

                self.output.append(self.policy_std)

            else:
                self.policy_std = tf.constant(self.exploration_variance, dtype='float32', shape=(self.num_actions,))

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean,
                                                                                       self.policy_std)

        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations = -tf.multiply(self.beta, self.entropy, name='entropy_regularization')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

            # calculate loss
            self.action_log_probs_wrt_policy = self.policy_distribution.log_prob(self.actions)
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.target = self.advantages
            self.loss = -tf.reduce_mean(self.action_log_probs_wrt_policy * self.advantages)
            tf.losses.add_loss(self.loss_weight[0] * self.loss)


class MeasurementsPredictionHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'future_measurements_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.num_measurements = tuning_parameters.env.measurements_size[0] \
            if tuning_parameters.env.measurements_size else 0
        self.num_prediction_steps = tuning_parameters.agent.num_predicted_steps_ahead
        self.multi_step_measurements_size = self.num_measurements * self.num_prediction_steps
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # This is almost exactly the same as Dueling Network but we predict the future measurements for each action
        # actions expectation tower (expectation stream) - E
        with tf.variable_scope("expectation_stream"):
            expectation_stream = tf.layers.dense(input_layer, 256, activation=tf.nn.elu)
            expectation_stream = tf.layers.dense(expectation_stream, self.multi_step_measurements_size)
            expectation_stream = tf.expand_dims(expectation_stream, axis=1)

        # action fine differences tower (action stream) - A
        with tf.variable_scope("action_stream"):
            action_stream = tf.layers.dense(input_layer, 256, activation=tf.nn.elu)
            action_stream = tf.layers.dense(action_stream, self.num_actions * self.multi_step_measurements_size)
            action_stream = tf.reshape(action_stream,
                                       (tf.shape(action_stream)[0], self.num_actions, self.multi_step_measurements_size))
            action_stream = action_stream - tf.reduce_mean(action_stream, reduction_indices=1, keep_dims=True)

        # merge to future measurements predictions
        self.output = tf.add(expectation_stream, action_stream, name='output')


class DNDQHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'dnd_q_values_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.DND_size = tuning_parameters.agent.dnd_size
        self.DND_key_error_threshold = tuning_parameters.agent.DND_key_error_threshold
        self.l2_norm_added_delta = tuning_parameters.agent.l2_norm_added_delta
        self.new_value_shift_coefficient = tuning_parameters.agent.new_value_shift_coefficient
        self.number_of_nn = tuning_parameters.agent.number_of_knn
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error
        self.tp = tuning_parameters

    def _build_module(self, input_layer):
        # DND based Q head
        from memories import differentiable_neural_dictionary

        if self.tp.checkpoint_restore_dir:
            self.DND = differentiable_neural_dictionary.load_dnd(self.tp.checkpoint_restore_dir)
        else:
            self.DND = differentiable_neural_dictionary.QDND(
                self.DND_size, input_layer.get_shape()[-1], self.num_actions, self.new_value_shift_coefficient,
                key_error_threshold=self.DND_key_error_threshold)

        # Retrieve info from DND dictionary
        self.action = tf.placeholder(tf.int8, [None], name="action")
        self.input = self.action
        result = tf.py_func(self.DND.query,
                            [input_layer, self.action, self.number_of_nn],
                            [tf.float64, tf.float64])
        self.dnd_embeddings = tf.to_float(result[0])
        self.dnd_values = tf.to_float(result[1])

        # DND calculation
        square_diff = tf.square(self.dnd_embeddings - tf.expand_dims(input_layer, 1))
        distances = tf.reduce_sum(square_diff, axis=2) + [self.l2_norm_added_delta]
        weights = 1.0 / distances
        normalised_weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
        self.output = tf.reduce_sum(self.dnd_values * normalised_weights, axis=1)


class NAFHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'naf_q_values_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.output_scale = np.max(tuning_parameters.env_instance.action_space_abs_range)
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # NAF
        self.action = tf.placeholder(tf.float32, [None, self.num_actions], name="action")
        self.input = self.action

        # V Head
        self.V = tf.layers.dense(input_layer, 1, name='V')

        # mu Head
        mu_unscaled = tf.layers.dense(input_layer, self.num_actions, activation=tf.nn.tanh, name='mu_unscaled')
        self.mu = tf.multiply(mu_unscaled, self.output_scale, name='mu')

        # A Head
        # l_vector is a vector that includes a lower-triangular matrix values
        self.l_vector = tf.layers.dense(input_layer, (self.num_actions * (self.num_actions + 1)) / 2, name='l_vector')

        # Convert l to a lower triangular matrix and exponentiate its diagonal

        i = 0
        columns = []
        for col in range(self.num_actions):
            start_row = col
            num_non_zero_elements = self.num_actions - start_row
            zeros_column_part = tf.zeros_like(self.l_vector[:, 0:start_row])
            diag_element = tf.expand_dims(tf.exp(self.l_vector[:, i]), 1)
            non_zeros_non_diag_column_part = self.l_vector[:, (i + 1):(i + num_non_zero_elements)]
            columns.append(tf.concat([zeros_column_part, diag_element, non_zeros_non_diag_column_part], axis=1))
            i += num_non_zero_elements
        self.L = tf.transpose(tf.stack(columns, axis=1), (0, 2, 1))

        # P = L*L^T
        self.P = tf.matmul(self.L, tf.transpose(self.L, (0, 2, 1)))

        # A = -1/2 * (u - mu)^T * P * (u - mu)
        action_diff = tf.expand_dims(self.action - self.mu, -1)
        a_matrix_form = -0.5 * tf.matmul(tf.transpose(action_diff, (0, 2, 1)), tf.matmul(self.P, action_diff))
        self.A = tf.reshape(a_matrix_form, [-1, 1])

        # Q Head
        self.Q = tf.add(self.V, self.A, name='Q')

        self.output = self.Q


class PPOHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'ppo_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.discrete_controls = tuning_parameters.env_instance.discrete_controls
        self.output_scale = np.max(tuning_parameters.env_instance.action_space_abs_range)

        # kl coefficient and its corresponding assignment operation and placeholder
        self.kl_coefficient = tf.Variable(tuning_parameters.agent.initial_kl_coefficient,
                                          trainable=False, name='kl_coefficient')
        self.kl_coefficient_ph = tf.placeholder('float', name='kl_coefficient_ph')
        self.assign_kl_coefficient = tf.assign(self.kl_coefficient, self.kl_coefficient_ph)

        self.kl_cutoff = 2*tuning_parameters.agent.target_kl_divergence
        self.high_kl_penalty_coefficient = tuning_parameters.agent.high_kl_penalty_coefficient
        self.clip_likelihood_ratio_using_epsilon = tuning_parameters.agent.clip_likelihood_ratio_using_epsilon
        self.use_kl_regularization = tuning_parameters.agent.use_kl_regularization
        self.beta = tuning_parameters.agent.beta_entropy

    def _build_module(self, input_layer):
        eps = 1e-15

        if self.discrete_controls:
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
        else:
            self.actions = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
        self.old_policy_mean = tf.placeholder(tf.float32, [None, self.num_actions], "old_policy_mean")
        self.old_policy_std = tf.placeholder(tf.float32, [None, self.num_actions], "old_policy_std")

        # Policy Head
        if self.discrete_controls:
            self.input = [self.actions, self.old_policy_mean]
            policy_values = tf.layers.dense(input_layer, self.num_actions)
            self.policy_mean = tf.nn.softmax(policy_values, name="policy")

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.contrib.distributions.Categorical(probs=self.policy_mean)
            self.old_policy_distribution = tf.contrib.distributions.Categorical(probs=self.old_policy_mean)

            self.output = self.policy_mean
        else:
            self.input = [self.actions, self.old_policy_mean, self.old_policy_std]
            self.policy_mean = tf.layers.dense(input_layer, self.num_actions, name='policy_mean')
            self.policy_logstd = tf.Variable(np.zeros((1, self.num_actions)), dtype='float32')
            self.policy_std = tf.tile(tf.exp(self.policy_logstd), [tf.shape(input_layer)[0], 1], name='policy_std')

            # define the distributions for the policy and the old policy
            self.policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean,
                                                                                       self.policy_std)
            self.old_policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.old_policy_mean,
                                                                                           self.old_policy_std)

            self.output = [self.policy_mean, self.policy_std]

        self.action_probs_wrt_policy = tf.exp(self.policy_distribution.log_prob(self.actions))
        self.action_probs_wrt_old_policy = tf.exp(self.old_policy_distribution.log_prob(self.actions))
        self.entropy = tf.reduce_mean(self.policy_distribution.entropy())

        # add kl divergence regularization
        self.kl_divergence = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.old_policy_distribution,
                                                                        self.policy_distribution))
        if self.use_kl_regularization:
            # no clipping => use kl regularization
            self.weighted_kl_divergence = tf.multiply(self.kl_coefficient, self.kl_divergence)
            self.regularizations = self.weighted_kl_divergence + self.high_kl_penalty_coefficient * \
                                                            tf.square(tf.maximum(0.0, self.kl_divergence - self.kl_cutoff))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

        # calculate surrogate loss
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.target = self.advantages
        self.likelihood_ratio = self.action_probs_wrt_policy / self.action_probs_wrt_old_policy
        if self.clip_likelihood_ratio_using_epsilon is not None:
            max_value = 1 + self.clip_likelihood_ratio_using_epsilon
            min_value = 1 - self.clip_likelihood_ratio_using_epsilon
            self.clipped_likelihood_ratio = tf.clip_by_value(self.likelihood_ratio, min_value, max_value)
            self.scaled_advantages = tf.minimum(self.likelihood_ratio * self.advantages,
                                                self.clipped_likelihood_ratio * self.advantages)
        else:
            self.scaled_advantages = self.likelihood_ratio * self.advantages
        # minus sign is in order to set an objective to minimize (we actually strive for maximizing the surrogate loss)
        self.surrogate_loss = -tf.reduce_mean(self.scaled_advantages)
        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations = -tf.multiply(self.beta, self.entropy, name='entropy_regularization')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

        self.loss = self.surrogate_loss
        tf.losses.add_loss(self.loss)


class PPOVHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'ppo_v_head'
        self.clip_likelihood_ratio_using_epsilon = tuning_parameters.agent.clip_likelihood_ratio_using_epsilon

    def _build_module(self, input_layer):
        self.old_policy_value = tf.placeholder(tf.float32, [None], "old_policy_values")
        self.input = [self.old_policy_value]
        self.output = tf.layers.dense(input_layer, 1, name='output',
                                            kernel_initializer=normalized_columns_initializer(1.0))
        self.target = self.total_return = tf.placeholder(tf.float32, [None], name="total_return")

        value_loss_1 = tf.square(self.output - self.target)
        value_loss_2 = tf.square(self.old_policy_value +
                                 tf.clip_by_value(self.output - self.old_policy_value,
                                                  -self.clip_likelihood_ratio_using_epsilon,
                                                  self.clip_likelihood_ratio_using_epsilon) - self.target)
        self.vf_loss = tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        self.loss = self.vf_loss
        tf.losses.add_loss(self.loss)


class CategoricalQHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'categorical_dqn_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.num_atoms = tuning_parameters.agent.atoms

    def _build_module(self, input_layer):
        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.input = [self.actions]

        values_distribution = tf.layers.dense(input_layer, self.num_actions * self.num_atoms)
        values_distribution = tf.reshape(values_distribution, (tf.shape(values_distribution)[0], self.num_actions, self.num_atoms))
        # softmax on atoms dimension
        self.output = tf.nn.softmax(values_distribution)

        # calculate cross entropy loss
        self.distributions = tf.placeholder(tf.float32, shape=(None, self.num_actions, self.num_atoms), name="distributions")
        self.target = self.distributions
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=values_distribution)
        tf.losses.add_loss(self.loss)


class QuantileRegressionQHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'quantile_regression_dqn_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.num_atoms = tuning_parameters.agent.atoms  # we use atom / quantile interchangeably
        self.huber_loss_interval = 1  # k

    def _build_module(self, input_layer):
        self.actions = tf.placeholder(tf.int32, [None, 2], name="actions")
        self.quantile_midpoints = tf.placeholder(tf.float32, [None, self.num_atoms], name="quantile_midpoints")
        self.input = [self.actions, self.quantile_midpoints]

        # the output of the head is the N unordered quantile locations {theta_1, ..., theta_N}
        quantiles_locations = tf.layers.dense(input_layer, self.num_actions * self.num_atoms)
        quantiles_locations = tf.reshape(quantiles_locations, (tf.shape(quantiles_locations)[0], self.num_actions, self.num_atoms))
        self.output = quantiles_locations

        self.quantiles = tf.placeholder(tf.float32, shape=(None, self.num_atoms), name="quantiles")
        self.target = self.quantiles

        # only the quantiles of the taken action are taken into account
        quantiles_for_used_actions = tf.gather_nd(quantiles_locations, self.actions)

        # reorder the output quantiles and the target quantiles as a preparation step for calculating the loss
        # the output quantiles vector and the quantile midpoints are tiled as rows of a NxN matrix (N = num quantiles)
        # the target quantiles vector is tiled as column of a NxN matrix
        theta_i = tf.tile(tf.expand_dims(quantiles_for_used_actions, -1), [1, 1, self.num_atoms])
        T_theta_j = tf.tile(tf.expand_dims(self.target, -2), [1, self.num_atoms, 1])
        tau_i = tf.tile(tf.expand_dims(self.quantile_midpoints, -1), [1, 1, self.num_atoms])

        # Huber loss of T(theta_j) - theta_i
        error = T_theta_j - theta_i
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.huber_loss_interval)
        huber_loss = self.huber_loss_interval * (abs_error - quadratic) + 0.5 * quadratic ** 2

        # Quantile Huber loss
        quantile_huber_loss = tf.abs(tau_i - tf.cast(error < 0, dtype=tf.float32)) * huber_loss

        # Quantile regression loss (the probability for each quantile is 1/num_quantiles)
        quantile_regression_loss = tf.reduce_sum(quantile_huber_loss) / float(self.num_atoms)
        self.loss = quantile_regression_loss
        tf.losses.add_loss(self.loss)
