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
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition


class PPOVHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'ppo_v_head'
        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.return_type = ActionProbabilities

    def _build_module(self, input_layer):
        self.old_policy_value = tf.placeholder(tf.float32, [None], "old_policy_values")
        self.input = [self.old_policy_value]
        self.output = self.dense_layer(1)(input_layer, name='output',
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

    def __str__(self):
        result = [
            "Dense (num outputs = 1)"
        ]
        return '\n'.join(result)
