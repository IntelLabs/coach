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

from typing import List, Union

from rl_coach.base_parameters import EmbedderScheme, NetworkComponentParameters


class InputEmbedderParameters(NetworkComponentParameters):
    def __init__(self, activation_function: str='relu', scheme: Union[List, EmbedderScheme]=EmbedderScheme.Medium,
                 batchnorm: bool=False, dropout_rate: float=0.0, name: str='embedder', input_rescaling=None,
                 input_offset=None, input_clipping=None, dense_layer=None, is_training=False):
        super().__init__(dense_layer=dense_layer)
        self.activation_function = activation_function
        self.scheme = scheme
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate

        if input_rescaling is None:
            input_rescaling = {'image': 255.0, 'vector': 1.0, 'tensor': 1.0}
        if input_offset is None:
            input_offset = {'image': 0.0, 'vector': 0.0, 'tensor': 0.0}

        self.input_rescaling = input_rescaling
        self.input_offset = input_offset
        self.input_clipping = input_clipping
        self.name = name
        self.is_training = is_training
