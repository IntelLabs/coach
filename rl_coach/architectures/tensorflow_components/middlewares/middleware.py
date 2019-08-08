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

import copy
from typing import Union, Tuple

import tensorflow as tf
from tensorflow import keras

from rl_coach.architectures.tensorflow_components.layers import BatchnormActivationDropout, convert_layer, Dense
from rl_coach.base_parameters import MiddlewareScheme, NetworkComponentParameters
from rl_coach.core_types import MiddlewareEmbedding


class Middleware(keras.layers.Layer):
    """
    A middleware embedder is the middle part of the network. It takes the embeddings from the input embedders,
    after they were aggregated in some method (for example, concatenation) and passes it through a neural network
    which can be customizable but shared between the heads of the network
    """
    def __init__(self,
                 activation_function=tf.nn.relu,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False,
                 dropout_rate: float = 0.0,
                 name="middleware_embedder",
                 is_training=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.middleware_layers = []
        # Dan manual fix self.name = name name is set in super().__init__ with self._init_set_name(name)
        #self.scheme = scheme
        self.return_type = MiddlewareEmbedding
        self.is_training = is_training

        for layer in self.schemes[scheme]:
            self.middleware_layers.extend([layer])
            if batchnorm:
                self.middleware_layers.extend([keras.layers.BatchNormalization()])
            if activation_function:
                self.middleware_layers.extend([keras.activations.get(activation_function)])
            if dropout_rate:
                self.middleware_layers.extend([keras.layers.Dropout(rate=dropout_rate)])

    def call(self, inputs, **kwargs):
        """
        Used for forward pass through middleware network.

        :param inputs: state embedding, of shape (batch_size, in_channels).
        :return: state middleware embedding, where shape is (batch_size, channels).
        """
        x = inputs
        for layer in self.middleware_layers:
            x = layer(x)
        return x


    def get_name(self) -> str:
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return self.name

    @property
    def schemes(self):
        raise NotImplementedError("Inheriting middleware must define schemes matching its allowed default "
                                  "configurations.")

    def __str__(self):
        result = [str(l) for l in self.layers_params]
        if self.layers_params:
            return '\n'.join(result)
        else:
            return 'No layers'
