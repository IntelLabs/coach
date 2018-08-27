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
from typing import Union, List

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.architecture import batchnorm_activation_dropout, Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware, MiddlewareParameters
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_FC_Embedding


class FCMiddlewareParameters(MiddlewareParameters):
    def __init__(self, activation_function='relu',
                 scheme: Union[List, MiddlewareScheme] = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout: bool = False,
                 name="middleware_fc_embedder", dense_layer=Dense):
        super().__init__(parameterized_class=FCMiddleware, activation_function=activation_function,
                         scheme=scheme, batchnorm=batchnorm, dropout=dropout, name=name, dense_layer=dense_layer)


class FCMiddleware(Middleware):
    def __init__(self, activation_function=tf.nn.relu,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout: bool = False,
                 name="middleware_fc_embedder", dense_layer=Dense):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout=dropout, scheme=scheme, name=name, dense_layer=dense_layer)
        self.return_type = Middleware_FC_Embedding
        self.layers = []

    def _build_module(self):
        self.layers.append(self.input)

        if isinstance(self.scheme, MiddlewareScheme):
            layers_params = self.schemes[self.scheme]
        else:
            layers_params = self.scheme
        for idx, layer_params in enumerate(layers_params):
            self.layers.append(
                layer_params(self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx))
            )

            self.layers.extend(batchnorm_activation_dropout(self.layers[-1], self.batchnorm,
                                                            self.activation_function, self.dropout,
                                                            self.dropout_rate, idx))

        self.output = self.layers[-1]

    @property
    def schemes(self):
        return {
            MiddlewareScheme.Empty:
                [],

            # ppo
            MiddlewareScheme.Shallow:
                [
                    self.dense_layer([64])
                ],

            # dqn
            MiddlewareScheme.Medium:
                [
                    self.dense_layer([512])
                ],

            MiddlewareScheme.Deep: \
                [
                    self.dense_layer([128]),
                    self.dense_layer([128]),
                    self.dense_layer([128])
                ]
        }

