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

from typing import List, Union, Tuple
import copy

import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import BatchnormActivationDropout, convert_layer, Dense
from rl_coach.base_parameters import EmbedderScheme, NetworkComponentParameters

from rl_coach.core_types import InputEmbedding
from rl_coach.utils import force_list


class InputEmbedder(object):
    """
    An input embedder is the first part of the network, which takes the input from the state and produces a vector
    embedding by passing it through a neural network. The embedder will mostly be input type dependent, and there
    can be multiple embedders in a single network
    """
    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 scheme: EmbedderScheme=None, batchnorm: bool=False, dropout_rate: float=0.0,
                 name: str= "embedder", input_rescaling=1.0, input_offset=0.0, input_clipping=None, dense_layer=Dense,
                 is_training=False):
        self.name = name
        self.input_size = input_size
        self.activation_function = activation_function
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.input = None
        self.output = None
        self.scheme = scheme
        self.return_type = InputEmbedding
        self.layers_params = []
        self.layers = []
        self.input_rescaling = input_rescaling
        self.input_offset = input_offset
        self.input_clipping = input_clipping
        self.dense_layer = dense_layer
        if self.dense_layer is None:
            self.dense_layer = Dense
        self.is_training = is_training

        # layers order is conv -> batchnorm -> activation -> dropout
        if isinstance(self.scheme, EmbedderScheme):
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

    def __call__(self, prev_input_placeholder: tf.placeholder=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wrapper for building the module graph including scoping and loss creation
        :param prev_input_placeholder: the input to the graph
        :return: the input placeholder and the output of the last layer
        """
        with tf.variable_scope(self.get_name()):
            if prev_input_placeholder is None:
                self.input = tf.placeholder("float", shape=[None] + self.input_size, name=self.get_name())
            else:
                self.input = prev_input_placeholder
            self._build_module()

        return self.input, self.output

    def _build_module(self) -> None:
        """
        Builds the graph of the module
        This method is called early on from __call__. It is expected to store the graph
        in self.output.
        :return: None
        """
        # NOTE: for image inputs, we expect the data format to be of type uint8, so to be memory efficient. we chose not
        #  to implement the rescaling as an input filters.observation.observation_filter, as this would have caused the
        #  input to the network to be float, which is 4x more expensive in memory.
        #  thus causing each saved transition in the memory to also be 4x more pricier.

        input_layer = self.input / self.input_rescaling
        input_layer -= self.input_offset
        # clip input using te given range
        if self.input_clipping is not None:
            input_layer = tf.clip_by_value(input_layer, self.input_clipping[0], self.input_clipping[1])

        self.layers.append(input_layer)

        for idx, layer_params in enumerate(self.layers_params):
            self.layers.extend(force_list(
                layer_params(input_layer=self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx),
                             is_training=self.is_training)
            ))

        self.output = tf.contrib.layers.flatten(self.layers[-1])

    @property
    def input_size(self) -> List[int]:
        return self._input_size

    @input_size.setter
    def input_size(self, value: Union[int, List[int]]):
        if isinstance(value, np.ndarray) or isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, int):
            value = [value]
        if not isinstance(value, list):
            raise ValueError((
                'input_size expected to be a list, found {value} which has type {type}'
            ).format(value=value, type=type(value)))
        self._input_size = value

    @property
    def schemes(self):
        raise NotImplementedError("Inheriting embedder must define schemes matching its allowed default "
                                  "configurations.")

    def get_name(self) -> str:
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return self.name

    def __str__(self):
        result = []
        if self.input_rescaling != 1.0 or self.input_offset != 0.0:
            result.append('Input Normalization (scale = {}, offset = {})'.format(self.input_rescaling, self.input_offset))
        result.extend([str(l) for l in self.layers_params])
        if self.layers_params:
            return '\n'.join(result)
        else:
            return 'No layers'
