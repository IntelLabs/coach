#
# Copyright (c) 2019 Intel Corporation
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
from rl_coach.core_types import VStateValue
from rl_coach.spaces import SpacesDefinition


class TD3VHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, initializer='xavier', output_bias_initializer=None):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'td3_v_values_head'
        self.return_type = VStateValue
        self.loss_type = []
        self.initializer = initializer
        self.loss = []
        self.output = []
        self.output_bias_initializer = output_bias_initializer

    def _build_module(self, input_layer):
        # Standard V Network
        q_outputs = []
        self.target = tf.placeholder(tf.float32, shape=(None, 1), name="q_networks_min_placeholder")

        for i in range(input_layer.shape[0]): # assuming that the actual size is 2, as there are two critic networks
            if self.initializer == 'normalized_columns':
                q_outputs.append(self.dense_layer(1)(input_layer[i], name='q_output_{}'.format(i + 1),
                                                     kernel_initializer=normalized_columns_initializer(1.0),
                                                     bias_initializer=self.output_bias_initializer),)
            elif self.initializer == 'xavier' or self.initializer is None:
                q_outputs.append(self.dense_layer(1)(input_layer[i], name='q_output_{}'.format(i + 1),
                                                     bias_initializer=self.output_bias_initializer))

            self.output.append(q_outputs[i])
            self.loss.append(tf.reduce_mean((self.target-q_outputs[i])**2))

        self.output.append(tf.reduce_min(q_outputs, axis=0))
        self.output.append(tf.reduce_mean(self.output[0]))
        self.loss = sum(self.loss)
        tf.losses.add_loss(self.loss)

    def __str__(self):
        result = [
            "Q1 Action-Value Stream",
            "\tDense (num outputs = 1)",
            "Q2 Action-Value Stream",
            "\tDense (num outputs = 1)",
            "Min (Q1, Q2)"
        ]
        return '\n'.join(result)
