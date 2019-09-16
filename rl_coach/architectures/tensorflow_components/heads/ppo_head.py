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


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters, DistributedTaskParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps


class PPOHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='tanh',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'ppo_head'
        self.return_type = ActionProbabilities

        # used in regular PPO
        self.use_kl_regularization = agent_parameters.algorithm.use_kl_regularization
        if self.use_kl_regularization:
            # kl coefficient and its corresponding assignment operation and placeholder
            self.kl_coefficient = tf.Variable(agent_parameters.algorithm.initial_kl_coefficient,
                                              trainable=False, name='kl_coefficient')
            self.kl_coefficient_ph = tf.compat.v1.placeholder('float', name='kl_coefficient_ph')
            self.assign_kl_coefficient = tf.compat.v1.assign(self.kl_coefficient, self.kl_coefficient_ph)
            self.kl_cutoff = 2 * agent_parameters.algorithm.target_kl_divergence
            self.high_kl_penalty_coefficient = agent_parameters.algorithm.high_kl_penalty_coefficient

        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.beta = agent_parameters.algorithm.beta_entropy

    def _build_module(self, input_layer):
        if isinstance(self.spaces.action, DiscreteActionSpace):
            self._build_discrete_net(input_layer, self.spaces.action)
        elif isinstance(self.spaces.action, BoxActionSpace):
            self._build_continuous_net(input_layer, self.spaces.action)
        else:
            raise ValueError("only discrete or continuous action spaces are supported for PPO")

        self.action_probs_wrt_policy = self.policy_distribution.log_prob(self.actions)
        self.action_probs_wrt_old_policy = self.old_policy_distribution.log_prob(self.actions)
        self.entropy = tf.reduce_mean(input_tensor=self.policy_distribution.entropy())

        # Used by regular PPO only
        # add kl divergence regularization
        self.kl_divergence = tf.reduce_mean(input_tensor=tf.compat.v1.distributions.kl_divergence(self.old_policy_distribution, self.policy_distribution))

        if self.use_kl_regularization:
            # no clipping => use kl regularization
            self.weighted_kl_divergence = tf.multiply(self.kl_coefficient, self.kl_divergence)
            self.regularizations += [self.weighted_kl_divergence + self.high_kl_penalty_coefficient * \
                                                tf.square(tf.maximum(0.0, self.kl_divergence - self.kl_cutoff))]

        # calculate surrogate loss
        self.advantages = tf.compat.v1.placeholder(tf.float32, [None], name="advantages")
        self.target = self.advantages
        # action_probs_wrt_old_policy != 0 because it is e^...
        self.likelihood_ratio = tf.exp(self.action_probs_wrt_policy - self.action_probs_wrt_old_policy)
        if self.clip_likelihood_ratio_using_epsilon is not None:
            self.clip_param_rescaler = tf.compat.v1.placeholder(tf.float32, ())
            self.input.append(self.clip_param_rescaler)
            max_value = 1 + self.clip_likelihood_ratio_using_epsilon * self.clip_param_rescaler
            min_value = 1 - self.clip_likelihood_ratio_using_epsilon * self.clip_param_rescaler
            self.clipped_likelihood_ratio = tf.clip_by_value(self.likelihood_ratio, min_value, max_value)
            self.scaled_advantages = tf.minimum(self.likelihood_ratio * self.advantages,
                                                self.clipped_likelihood_ratio * self.advantages)
        else:
            self.scaled_advantages = self.likelihood_ratio * self.advantages
        # minus sign is in order to set an objective to minimize (we actually strive for maximizing the surrogate loss)
        self.surrogate_loss = -tf.reduce_mean(input_tensor=self.scaled_advantages)
        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(input_tensor=self.policy_distribution.entropy())
                self.regularizations += [-tf.multiply(self.beta, self.entropy, name='entropy_regularization')]

        self.loss = self.surrogate_loss
        tf.compat.v1.losses.add_loss(self.loss)

    def _build_discrete_net(self, input_layer, action_space):
        num_actions = len(action_space.actions)
        self.actions = tf.compat.v1.placeholder(tf.int32, [None], name="actions")

        self.old_policy_mean = tf.compat.v1.placeholder(tf.float32, [None, num_actions], "old_policy_mean")
        self.old_policy_std = tf.compat.v1.placeholder(tf.float32, [None, num_actions], "old_policy_std")

        # Policy Head
        self.input = [self.actions, self.old_policy_mean]
        policy_values = self.dense_layer(num_actions)(input_layer, name='policy_fc')
        self.policy_mean = tf.nn.softmax(policy_values, name="policy")

        # define the distributions for the policy and the old policy
        self.policy_distribution = tf.compat.v1.random.categorical(probs=self.policy_mean)

        self.old_policy_distribution = tf.compat.v1.random.categorical(probs=self.old_policy_mean)


        self.output = self.policy_mean

    def _build_continuous_net(self, input_layer, action_space):
        num_actions = action_space.shape[0]
        self.actions = tf.compat.v1.placeholder(tf.float32, [None, num_actions], name="actions")

        self.old_policy_mean = tf.compat.v1.placeholder(tf.float32, [None, num_actions], "old_policy_mean")
        self.old_policy_std = tf.compat.v1.placeholder(tf.float32, [None, num_actions], "old_policy_std")

        self.input = [self.actions, self.old_policy_mean, self.old_policy_std]
        self.policy_mean = self.dense_layer(num_actions)(input_layer, name='policy_mean',
                                           kernel_initializer=normalized_columns_initializer(0.01))

        # for local networks in distributed settings, we need to move variables we create manually to the
        # tf.GraphKeys.LOCAL_VARIABLES collection, since the variable scope custom getter which is set in
        # Architecture does not apply to them
        if self.is_local and isinstance(self.ap.task_parameters, DistributedTaskParameters):
            self.policy_logstd = tf.Variable(np.zeros((1, num_actions)), dtype='float32',
                                             collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES], name="policy_log_std")
        else:
            self.policy_logstd = tf.Variable(np.zeros((1, num_actions)), dtype='float32', name="policy_log_std")

        self.policy_std = tf.tile(tf.exp(self.policy_logstd), [tf.shape(input=input_layer)[0], 1], name='policy_std')

        # define the distributions for the policy and the old policy
        self.policy_distribution = tfp.distributions.MultivariateNormalDiag(self.policy_mean,
                                                                                   self.policy_std + eps)
        self.old_policy_distribution = tfp.distributions.MultivariateNormalDiag(self.old_policy_mean,
                                                                                       self.old_policy_std + eps)

        # self.policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean, self.policy_std + eps)
        # self.old_policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.old_policy_mean, self.old_policy_std + eps)

        self.output = [self.policy_mean, self.policy_std]

    def __str__(self):
        action_head_mean_result = []
        if isinstance(self.spaces.action, DiscreteActionSpace):
            # create a discrete action network (softmax probabilities output)
            action_head_mean_result.append("Dense (num outputs = {})".format(len(self.spaces.action.actions)))
            action_head_mean_result.append("Softmax")
        elif isinstance(self.spaces.action, BoxActionSpace):
            # create a continuous action network (bounded mean and stdev outputs)
            action_head_mean_result.append("Dense (num outputs = {})".format(self.spaces.action.shape))

        return '\n'.join(action_head_mean_result)
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


from typing import List, Tuple, Union
from types import ModuleType

import math
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace
from rl_coach.utils import eps


from rl_coach.architectures.tensorflow_components.heads.head import Head



LOSS_OUT_TYPE_KL = 'kl_divergence'
LOSS_OUT_TYPE_ENTROPY = 'entropy'
LOSS_OUT_TYPE_LIKELIHOOD_RATIO = 'likelihood_ratio'
LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO = 'clipped_likelihood_ratio'

from tensorflow import keras
from tensorflow import Tensor


class ContinuousPPOHead(keras.layers.Layer):
    def __init__(self, num_actions: int) -> None:
        """
        Head block for Continuous Proximal Policy Optimization, to calculate probabilities for each action given
        middleware representation of the environment state.

        :param num_actions: number of actions in action space.
        """
        super(ContinuousPPOHead, self).__init__()

        self.dense = tf.keras.layers.Dense(units=num_actions)
        # all samples (across batch, and time step) share the same covariance, which is learnt,
        # but since we assume the action probability variables are independent,
        # only the diagonal entries of the covariance matrix are specified.
        self.log_std = np.zeros(shape=(num_actions, 1))


    def call(self, inputs) -> Tuple[Tensor, Tensor]:
        """
        Used for forward pass through head network.

        :param inputs: middleware state representation,
            of shape (batch_size, in_channels) or
            of shape (batch_size, time_step, in_channels).
        :return: batch of probabilities for each action,
            of shape (batch_size, action_mean) or
            of shape (batch_size, time_step, action_mean).
        """
        policy_means = self.dense(inputs)
        policy_std = np.exp(self.log_std)
        return policy_means, policy_std



class PPOHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='tanh',
                 dense_layer: None=None) -> None:
        """
        Head block for Proximal Policy Optimization, to calculate probabilities for each action given middleware
        representation of the environment state.

        :param agent_parameters: containing algorithm parameters such as clip_likelihood_ratio_using_epsilon
            and beta_entropy.
        :param spaces: containing action spaces used for defining size of network output.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super().__init__(agent_parameters, spaces, network_name, head_type_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.return_type = ActionProbabilities

        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.beta = agent_parameters.algorithm.beta_entropy
        self.use_kl_regularization = agent_parameters.algorithm.use_kl_regularization
        if self.use_kl_regularization:
            self.initial_kl_coefficient = agent_parameters.algorithm.initial_kl_coefficient
            self.kl_cutoff = 2 * agent_parameters.algorithm.target_kl_divergence
            self.high_kl_penalty_coefficient = agent_parameters.algorithm.high_kl_penalty_coefficient
        else:
            self.initial_kl_coefficient, self.kl_cutoff, self.high_kl_penalty_coefficient = (None, None, None)
        self._loss = []

        if isinstance(self.spaces.action, BoxActionSpace):
            self.net = ContinuousPPOHead(num_actions=self.spaces.action.shape[0])
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for PPO.")

    def call(self, inputs, ** kwargs):
        """
        :param inputs: middleware embedding
        :return: policy parameters/probabilities
        """
        return self.net(inputs)


    @property
    def kl_divergence(self):
        return self.head_type_idx, LOSS_OUT_TYPE_KL

    @property
    def entropy(self):
        return self.head_type_idx, LOSS_OUT_TYPE_ENTROPY

    @property
    def likelihood_ratio(self):
        return self.head_type_idx, LOSS_OUT_TYPE_LIKELIHOOD_RATIO

    @property
    def clipped_likelihood_ratio(self):
        return self.head_type_idx, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO

    def assign_kl_coefficient(self, kl_coefficient: float) -> None:
        self._loss[0]