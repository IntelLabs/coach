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

from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import Embedding
from rl_coach.spaces import SpacesDefinition


class RNDHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, is_local: bool = True):
        super().__init__(agent_parameters, spaces, network_name, head_idx, is_local)
        self.name = 'rnd_values_head'
        self.return_type = Embedding

        self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        self.output = self.dense_layer(512)(input_layer, name='output')
