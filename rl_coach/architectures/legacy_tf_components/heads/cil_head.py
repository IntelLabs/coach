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
from rl_coach.utils import force_list


class RegressionHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, scheme=[Dense(256), Dense(256)]):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'regression_head'
        self.scheme = scheme
        self.layers = []
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = self.spaces.action.shape[0]
        elif isinstance(self.spaces.action, DiscreteActionSpace):
            self.num_actions = len(self.spaces.action.actions)
        self.return_type = QActionStateValue
        if agent_parameters.network_wrappers[self.network_name].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        self.layers.append(input_layer)
        for idx, layer_params in enumerate(self.scheme):
            self.layers.extend(force_list(
                layer_params(input_layer=self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx))
            ))

        self.layers.append(self.dense_layer(self.num_actions)(self.layers[-1], name='output'))
        self.output = self.layers[-1]

    def __str__(self):
        result = []
        for layer in self.layers:
            result.append(str(layer))
        return '\n'.join(result)

