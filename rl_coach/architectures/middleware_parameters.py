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

from typing import List, Type, Union

from rl_coach.base_parameters import MiddlewareScheme, NetworkComponentParameters


class MiddlewareParameters(NetworkComponentParameters):
    def __init__(self, parameterized_class_name: str,
                 activation_function: str='relu', scheme: Union[List, MiddlewareScheme]=MiddlewareScheme.Medium,
                 batchnorm: bool=False, dropout_rate: float=0.0, name='middleware', dense_layer=None, is_training=False):
        super().__init__(dense_layer=dense_layer)
        self.activation_function = activation_function
        self.scheme = scheme
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.name = name
        self.is_training = is_training
        self.parameterized_class_name = parameterized_class_name


class FCMiddlewareParameters(MiddlewareParameters):
    def __init__(self, activation_function='relu',
                 scheme: Union[List, MiddlewareScheme] = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_fc_embedder", dense_layer=None, is_training=False):
        super().__init__(parameterized_class_name="FCMiddleware", activation_function=activation_function,
                         scheme=scheme, batchnorm=batchnorm, dropout_rate=dropout_rate, name=name, dense_layer=dense_layer,
                         is_training=is_training)


class LSTMMiddlewareParameters(MiddlewareParameters):
    def __init__(self, activation_function='relu', number_of_lstm_cells=256,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_lstm_embedder", dense_layer=None, is_training=False):
        super().__init__(parameterized_class_name="LSTMMiddleware", activation_function=activation_function,
                         scheme=scheme, batchnorm=batchnorm, dropout_rate=dropout_rate, name=name, dense_layer=dense_layer,
                         is_training=is_training)
        self.number_of_lstm_cells = number_of_lstm_cells