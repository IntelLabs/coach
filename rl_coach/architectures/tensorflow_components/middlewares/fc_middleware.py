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

from rl_coach.architectures.tensorflow_components.layers import batchnorm_activation_dropout, Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_FC_Embedding
from rl_coach.utils import force_list


class FCMiddleware(Middleware):
    def __init__(self, activation_function=tf.nn.relu,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_fc_embedder", dense_layer=Dense, is_training=False):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout_rate=dropout_rate, scheme=scheme, name=name, dense_layer=dense_layer,
                         is_training=is_training)
        self.return_type = Middleware_FC_Embedding
        self.layers = []

    def _build_module(self):
        self.layers.append(self.input)

        for idx, layer_params in enumerate(self.layers_params):
            self.layers.extend(force_list(
                layer_params(self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx),
                             is_training=self.is_training)
            ))

        self.output = self.layers[-1]

    @property
    def schemes(self):
        return {
            MiddlewareScheme.Empty:
                [],

            # ppo
            MiddlewareScheme.Shallow:
                [
                    self.dense_layer(64)
                ],

            # dqn
            MiddlewareScheme.Medium:
                [
                    self.dense_layer(512)
                ],

            MiddlewareScheme.Deep: \
                [
                    self.dense_layer(128),
                    self.dense_layer(128),
                    self.dense_layer(128)
                ]
        }

