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

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.exploration_policies.continuous_entropy import ContinuousEntropyParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, CompoundActionSpace
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps, indent_string


class ACERPolicyHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'acer_policy_head'
        self.return_type = ActionProbabilities
        self.beta = agent_parameters.algorithm.beta_entropy

    def _build_module(self, input_layer):
        if isinstance(self.spaces.action, DiscreteActionSpace):
            self._build_discrete_net(input_layer, self.spaces.action)
        else:
            raise ValueError("only discrete action spaces are supported for ACER")

        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
                self.regularizations = -tf.multiply(self.beta, self.entropy, name='entropy_regularization')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

            # Truncated importance sampling with bias corrections
            importance_weight_i = tf.placeholder(tf.float32, [None],
                                                 name='{}_importance_weight_i'.format(self.get_name()))
            self.importance_weight.append(importance_weight_i)
            importance_weight = tf.placeholder(tf.float32, [None, self.num_actions],
                                               name='{}_importance_weight'.format(self.get_name()))
            self.importance_weight.append(importance_weight)
            advantages = tf.placeholder(tf.float32, [None],
                                        name='{}_advantages'.format(self.get_name()))
            self.target.append(advantages)
            advantages_bc = tf.placeholder(tf.float32, [None, self.num_actions],
                                           name='{}_advantages_bc'.format(self.get_name()))
            self.target.append(advantages_bc)

            action_log_probs_wrt_policy = self.policy_distribution.log_prob(self.actions)
            self.prob_loss = -tf.reduce_mean(action_log_probs_wrt_policy
                                             * advantages
                                             * tf.minimum(self.ap.algorithm.importance_weight_truncation,
                                                          importance_weight_i))

            log_probs_wrt_policy = tf.log(self.policy_probs + eps)
            gain_bc = tf.reduce_sum(log_probs_wrt_policy
                                    * advantages_bc
                                    * tf.nn.relu(1.0 - (self.ap.algorithm.importance_weight_truncation / (importance_weight + eps)))
                                    * tf.stop_gradient(self.policy_probs),
                                    axis=1)
            self.bc_loss = -tf.reduce_mean(gain_bc)

            self.loss = self.prob_loss + self.bc_loss
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

    def calculate_trust_region_gradients(self, weights_placeholders):
        grads = tf.gradients(self.policy_probs, weights_placeholders, self.policy_grads_wrt_network_output)
        return [g for g in grads if g is not None]
