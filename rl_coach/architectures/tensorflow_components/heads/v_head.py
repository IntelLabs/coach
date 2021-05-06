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

from rl_coach.architectures.tensorflow_components.layers import Dense, SchemeBuilder
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import VStateValue
from rl_coach.spaces import SpacesDefinition


class VHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, initializer='normalized_columns', output_bias_initializer=None):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'v_values_head'
        self.return_type = VStateValue

        if agent_parameters.network_wrappers[self.network_name.split('/')[0]].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

        self.initializer = initializer
        self.output_bias_initializer = output_bias_initializer

    def _build_module(self, input_layer):
        # Standard V Network
        if self.initializer == 'normalized_columns':
            self.output = self.dense_layer(1)(input_layer, name='output',
                                              kernel_initializer=normalized_columns_initializer(1.0),
                                              bias_initializer=self.output_bias_initializer)
        elif self.initializer == 'xavier' or self.initializer is None:
            self.output = self.dense_layer(1)(input_layer, name='output',
                                              bias_initializer=self.output_bias_initializer)

    def __str__(self):
        result = [
            "Dense (num outputs = 1)"
        ]
        return '\n'.join(result)


class VHeadWithPreDense(VHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, initializer='normalized_columns', output_bias_initializer=None,
                 pre_dense_sizes=None, pre_dense_activation_function='relu'):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer, initializer, output_bias_initializer)
        self.name = 'v_values_w_pre_dense_head'
        pre_dense_sizes = pre_dense_sizes or []
        self.pre_dense_builder = SchemeBuilder([self.dense_layer(size) for size in pre_dense_sizes],
                                               pre_dense_activation_function)
        self.pre_dense_layers = None

    def _build_module(self, input_layer):
        self.pre_dense_layers = self.pre_dense_builder.build_scheme(input_layer, name_prefix='pre_dense')
        super(VHeadWithPreDense, self)._build_module(self.pre_dense_layers[-1])

    def __str__(self):
        result = [str(layer) for layer in self.pre_dense_builder.layers_params]
        result.append(super(VHeadWithPreDense, self).__str__())
        return '\n'.join(result)
