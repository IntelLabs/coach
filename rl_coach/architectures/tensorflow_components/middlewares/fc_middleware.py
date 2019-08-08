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
from tensorflow import keras

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_FC_Embedding
from rl_coach.utils import force_list


class FCMiddleware(Middleware):
    def __init__(self,
                 activation_function=tf.nn.relu,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False,
                 dropout_rate: float = 0.0,
                 name="middleware_fc_embedder",
                 is_training=False,
                 num_streams: int = 1):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout_rate=dropout_rate, scheme=scheme, name=name,
                         is_training=is_training)
        self.return_type = Middleware_FC_Embedding

        assert(isinstance(num_streams, int) and num_streams >= 1)
        self.num_streams = num_streams

    @property
    def schemes(self):
        return {
            MiddlewareScheme.Empty:
                [],

            # Use for PPO
            MiddlewareScheme.Shallow:
                [
                    keras.layers.Dense(64)
                ],

            # Use for DQN
            MiddlewareScheme.Medium:
                [
                    keras.layers.Dense(512)
                ],

            MiddlewareScheme.Deep: \
                [
                    keras.layers.Dense(128),
                    keras.layers.Dense(128),
                    keras.layers.Dense(128)
                ]
        }

    def __str__(self):
        stream = [str(l) for l in self.layers_params]
        if self.layers_params:
            if self.num_streams > 1:
                stream = [''] + ['\t' + l for l in stream]
                result = stream * self.num_streams
                result[0::len(stream)] = ['Stream {}'.format(i) for i in range(self.num_streams)]
            else:
                result = stream
            return '\n'.join(result)
        else:
            return 'No layers'
