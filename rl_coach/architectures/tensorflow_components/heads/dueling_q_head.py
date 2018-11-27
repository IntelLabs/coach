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
from rl_coach.architectures.tensorflow_components.heads.q_head import QHead
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class DuelingQHead(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'dueling_q_values_head'

    def _build_module(self, input_layer):
        # state value tower - V
        with tf.variable_scope("state_value"):
            self.state_value = self.dense_layer(512)(input_layer, activation=self.activation_function, name='fc1')
            self.state_value = self.dense_layer(1)(self.state_value, name='fc2')

        # action advantage tower - A
        with tf.variable_scope("action_advantage"):
            self.action_advantage = self.dense_layer(512)(input_layer, activation=self.activation_function, name='fc1')
            self.action_advantage = self.dense_layer(self.num_actions)(self.action_advantage, name='fc2')
            self.action_mean = tf.reduce_mean(self.action_advantage, axis=1, keepdims=True)
            self.action_advantage = self.action_advantage - self.action_mean

        # merge to state-action value function Q
        self.output = tf.add(self.state_value, self.action_advantage, name='output')

    def __str__(self):
        result = [
            "State Value Stream - V",
            "\tDense (num outputs = 512)",
            "\tDense (num outputs = 1)",
            "Action Advantage Stream - A",
            "\tDense (num outputs = 512)",
            "\tDense (num outputs = {})".format(self.num_actions),
            "\tSubtract(A, Mean(A))".format(self.num_actions),
            "Add (V, A)"
        ]
        return '\n'.join(result)
