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

from rl_coach.architectures.tensorflow_components.layers import BatchnormActivationDropout, convert_layer, Dense
from rl_coach.base_parameters import MiddlewareScheme, NetworkComponentParameters
from rl_coach.core_types import MiddlewareEmbedding


class Middleware(object):
    """
    A middleware embedder is the middle part of the network. It takes the embeddings from the input embedders,
    after they were aggregated in some method (for example, concatenation) and passes it through a neural network
    which can be customizable but shared between the heads of the network
    """
    def __init__(self, activation_function=tf.nn.relu,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0, name="middleware_embedder", dense_layer=Dense,
                 is_training=False):
        self.name = name
        self.input = None
        self.output = None
        self.activation_function = activation_function
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.scheme = scheme
        self.return_type = MiddlewareEmbedding
        self.dense_layer = dense_layer
        if self.dense_layer is None:
            self.dense_layer = Dense
        self.is_training = is_training

        # layers order is conv -> batchnorm -> activation -> dropout
        if isinstance(self.scheme, MiddlewareScheme):
            self.layers_params = copy.copy(self.schemes[self.scheme])
        else:
            # if scheme is specified directly, convert to TF layer if it's not a callable object
            # NOTE: if layer object is callable, it must return a TF tensor when invoked
            self.layers_params = [convert_layer(l) for l in copy.copy(self.scheme)]

        # we allow adding batchnorm, dropout or activation functions after each layer.
        # The motivation is to simplify the transition between a network with batchnorm and a network without
        # batchnorm to a single flag (the same applies to activation function and dropout)
        if self.batchnorm or self.activation_function or self.dropout_rate > 0:
            for layer_idx in reversed(range(len(self.layers_params))):
                self.layers_params.insert(layer_idx+1,
                                          BatchnormActivationDropout(batchnorm=self.batchnorm,
                                                                     activation_function=self.activation_function,
                                                                     dropout_rate=self.dropout_rate))

    def __call__(self, input_layer: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wrapper for building the module graph including scoping and loss creation
        :param input_layer: the input to the graph
        :return: the input placeholder and the output of the last layer
        """
        with tf.variable_scope(self.get_name()):
            self.input = input_layer
            self._build_module()

        return self.input, self.output

    def _build_module(self) -> None:
        """
        Builds the graph of the module
        This method is called early on from __call__. It is expected to store the graph
        in self.output.
        :param input_layer: the input to the graph
        :return: None
        """
        pass

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
