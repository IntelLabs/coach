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
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace
from tensorflow.python.ops.losses.losses_impl import Reduction


class ACERQHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'acer_q_values_head'
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = 1
        elif isinstance(self.spaces.action, DiscreteActionSpace):
            self.num_actions = len(self.spaces.action.actions)
        self.return_type = QActionStateValue

    def _build_module(self, input_layer):
        self.output = self.dense_layer(self.num_actions)(input_layer, name='output')

        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.input = self.actions
        target = tf.placeholder(tf.float32, [None], 'acer_q_values_head_target'.format(self.get_name()))
        self.target = target

        idx_flattened = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.actions
        Q_i = tf.gather(tf.reshape(self.output, [-1]), idx_flattened)

        if self.is_local:
            self.loss = tf.losses.mean_squared_error(self.target, Q_i,
                                                     scope=self.get_name(), reduction=Reduction.MEAN,
                                                     loss_collection=None)
            tf.losses.add_loss(self.loss)

    def __str__(self):
        result = [
            "Dense (num outputs = {})".format(self.num_actions)
        ]
        return '\n'.join(result)


