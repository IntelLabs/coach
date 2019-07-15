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

from rl_coach.architectures.tensorflow_components.heads import VHead
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class DDPGVHead(VHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, initializer='normalized_columns', output_bias_initializer=None):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer, initializer=initializer,
                         output_bias_initializer=output_bias_initializer)

    def _build_module(self, input_layer):
        super()._build_module(input_layer)
        self.output = [self.output, tf.reduce_mean(self.output)]

    def __str__(self):
        result = [
            "Dense (num outputs = 1)"
        ]
        return '\n'.join(result)
