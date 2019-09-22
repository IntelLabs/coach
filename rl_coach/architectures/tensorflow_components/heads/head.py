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


import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition

# Used to initialize weights for policy and value output layers
# def normalized_columns_initializer(std=1.0):
#     def _initializer(shape, dtype=None, partition_info=None):
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#     return _initializer


class Head(keras.layers.Layer):
    def __init__(self, agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='relu',
                 dense_layer: None=None):
        """
        A head is the final part of the network. It takes the embedding from the middleware embedder and passes it
        through a neural network to produce the output of the network. There can be multiple heads in a network, and
        each one has an assigned loss function. The heads are algorithm dependent.

        :param agent_parameters: containing algorithm parameters such as clip_likelihood_ratio_using_epsilon
            and beta_entropy.
        :param spaces: containing action spaces used for defining size of network output.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super(Head, self).__init__()
        self.head_type_idx = head_type_idx
        self.network_name = network_name
        self.loss_weight = loss_weight
        self.is_local = is_local
        self.ap = agent_parameters
        self.spaces = spaces
        self.return_type = None
        self.activation_function = activation_function
        self.dense_layer = dense_layer
        self._num_outputs = None

    @property
    def num_outputs(self):
        """ Returns number of outputs that forward() call will return

        :return:
        """
        assert self._num_outputs is not None, 'must call forward() once to configure number of outputs'
        return self._num_outputs


    def forward(self, *args):
        """
        Override forward() so that number of outputs can be automatically set
        """
        outputs = super(Head, self).forward(*args)
        num_outputs = len(outputs)
        if self._num_outputs is None:
            self._num_outputs = num_outputs
        else:
            assert self._num_outputs == num_outputs, 'Number of outputs cannot change ({} != {})'.format(
                self._num_outputs, num_outputs)
        assert self._num_outputs == len(self.loss().input_schema.head_outputs)
        return outputs

    #def compute_output_shape(self):

    #def call(self, inputs, **kwargs):

