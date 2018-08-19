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

from rl_coach.architectures.tensorflow_components.heads.head import HeadParameters
from rl_coach.architectures.tensorflow_components.heads.q_head import QHead
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class DuelingQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='dueling_q_head_params'):
        super().__init__(parameterized_class=DuelingQHead, activation_function=activation_function, name=name)


class DuelingQHead(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu'):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function)
        self.name = 'dueling_q_values_head'

    def _build_module(self, input_layer):
        # state value tower - V
        with tf.variable_scope("state_value"):
            state_value = tf.layers.dense(input_layer, 512, activation=self.activation_function, name='fc1')
            state_value = tf.layers.dense(state_value, 1, name='fc2')
            # state_value = tf.expand_dims(state_value, axis=-1)

        # action advantage tower - A
        with tf.variable_scope("action_advantage"):
            action_advantage = tf.layers.dense(input_layer, 512, activation=self.activation_function, name='fc1')
            action_advantage = tf.layers.dense(action_advantage, self.num_actions, name='fc2')
            action_advantage = action_advantage - tf.reduce_mean(action_advantage)

        # merge to state-action value function Q
        self.output = tf.add(state_value, action_advantage, name='output')
