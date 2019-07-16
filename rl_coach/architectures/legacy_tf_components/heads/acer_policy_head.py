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

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import DiscreteActionSpace
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps


class ACERPolicyHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'acer_policy_head'
        self.return_type = ActionProbabilities
        self.beta = None
        self.action_penalty = None

        # a scalar weight that penalizes low entropy values to encourage exploration
        if hasattr(agent_parameters.algorithm, 'beta_entropy'):
            # we set the beta value as a tf variable so it can be updated later if needed
            self.beta = tf.Variable(float(agent_parameters.algorithm.beta_entropy),
                                    trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.beta_placeholder = tf.placeholder('float')
            self.set_beta = tf.assign(self.beta, self.beta_placeholder)

    def _build_module(self, input_layer):
        if isinstance(self.spaces.action, DiscreteActionSpace):
            # create a discrete action network (softmax probabilities output)
            self._build_discrete_net(input_layer, self.spaces.action)
        else:
            raise ValueError("only discrete action spaces are supported for ACER")

        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations += [-tf.multiply(self.beta, self.entropy, name='entropy_regularization')]

            # Truncated importance sampling with bias corrections
            importance_sampling_weight = tf.placeholder(tf.float32, [None, self.num_actions],
                                                        name='{}_importance_sampling_weight'.format(self.get_name()))
            self.input.append(importance_sampling_weight)
            importance_sampling_weight_i = tf.placeholder(tf.float32, [None],
                                                          name='{}_importance_sampling_weight_i'.format(self.get_name()))
            self.input.append(importance_sampling_weight_i)

            V_values = tf.placeholder(tf.float32, [None], name='{}_V_values'.format(self.get_name()))
            self.target.append(V_values)
            Q_values = tf.placeholder(tf.float32, [None, self.num_actions], name='{}_Q_values'.format(self.get_name()))
            self.input.append(Q_values)
            Q_retrace = tf.placeholder(tf.float32, [None], name='{}_Q_retrace'.format(self.get_name()))
            self.input.append(Q_retrace)

            action_log_probs_wrt_policy = self.policy_distribution.log_prob(self.actions)
            self.probability_loss = -tf.reduce_mean(action_log_probs_wrt_policy
                                                    * (Q_retrace - V_values)
                                                    * tf.minimum(self.ap.algorithm.importance_weight_truncation,
                                                                 importance_sampling_weight_i))

            log_probs_wrt_policy = tf.log(self.policy_probs + eps)
            bias_correction_gain = tf.reduce_sum(log_probs_wrt_policy
                                                 * (Q_values - tf.expand_dims(V_values, 1))
                                                 * tf.nn.relu(1.0 - (self.ap.algorithm.importance_weight_truncation
                                                                     / (importance_sampling_weight + eps)))
                                                 * tf.stop_gradient(self.policy_probs),
                                                 axis=1)
            self.bias_correction_loss = -tf.reduce_mean(bias_correction_gain)

            self.loss = self.probability_loss + self.bias_correction_loss
            tf.losses.add_loss(self.loss)

            # Trust region
            batch_size = tf.to_float(tf.shape(input_layer)[0])
            average_policy = tf.placeholder(tf.float32, [None, self.num_actions],
                                            name='{}_average_policy'.format(self.get_name()))
            self.input.append(average_policy)
            average_policy_distribution = tf.contrib.distributions.Categorical(probs=(average_policy + eps))
            self.kl_divergence = tf.reduce_mean(tf.distributions.kl_divergence(average_policy_distribution,
                                                                               self.policy_distribution))
            if self.ap.algorithm.use_trust_region_optimization:
                @tf.custom_gradient
                def trust_region_layer(x):
                    def grad(g):
                        g = - g * batch_size
                        k = - average_policy / (self.policy_probs + eps)
                        adj = tf.nn.relu(
                            (tf.reduce_sum(k * g, axis=1) - self.ap.algorithm.max_KL_divergence)
                            / (tf.reduce_sum(tf.square(k), axis=1) + eps))
                        g = g - tf.expand_dims(adj, 1) * k
                        return - g / batch_size
                    return tf.identity(x), grad
                self.output = trust_region_layer(self.output)

    def _build_discrete_net(self, input_layer, action_space):
        self.num_actions = len(action_space.actions)
        self.actions = tf.placeholder(tf.int32, [None], name='{}_actions'.format(self.get_name()))
        self.input.append(self.actions)

        policy_values = self.dense_layer(self.num_actions)(input_layer, name='fc')
        self.policy_probs = tf.nn.softmax(policy_values, name='{}_policy'.format(self.get_name()))

        # (the + eps is to prevent probability 0 which will cause the log later on to be -inf)
        self.policy_distribution = tf.contrib.distributions.Categorical(probs=(self.policy_probs + eps))
        self.output = self.policy_probs
