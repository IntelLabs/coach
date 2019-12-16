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

from typing import Dict
import tensorflow as tf
from rl_coach.architectures.layers import Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_FC_Embedding

"""
Module that defines the fully-connected middleware class
"""


class FCMiddleware(Middleware):
    """
    FCMiddleware or Fully-Connected Middleware can be used in the middle part of the network. It takes the
    embeddings from the input embedders, after they were aggregated in some method (for example, concatenation)
    and passes it through a neural network  which can be customizable but shared between the heads of the network.

    :param params: parameters object containing batchnorm, activation_function and dropout properties.
    """
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
    def schemes(self) -> Dict:
        """
       Schemes are the pre-defined network architectures of various depths and complexities that can be used for the
       Middleware. Are used to create Block when FCMiddleware is initialised.

       :return: dictionary of schemes, with key of type MiddlewareScheme enum and value being list of Tensorflow layers.
       """
        return {
            MiddlewareScheme.Empty:
                [],

            # Use for PPO
            MiddlewareScheme.Shallow:
                [
                    Dense(64),
                ],

            # Use for DQN
            MiddlewareScheme.Medium:
                [
                    Dense(512),
                ],

            MiddlewareScheme.Deep: \
                [
                    Dense(128),
                    Dense(128),
                    Dense(128)
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
