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

from typing import List

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Conv2d, Dense
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedder
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.core_types import InputTensorEmbedding


class TensorEmbedder(InputEmbedder):
    """
    A tensor embedder is an input embedder that takes a tensor with arbitrary dimension and produces a vector
    embedding by passing it through a neural network. An example is video data or 3D image data (i.e. 4D tensors)
    or other type of data that is more than 1 dimension (i.e. not vector) but is not an image.

    NOTE: There are no pre-defined schemes for tensor embedder. User must define a custom scheme by passing
    a callable object as InputEmbedderParameters.scheme when defining the respective preset. This callable
    object must accept a single input, the normalized observation, and return a Tensorflow symbol which
    will calculate an embedding vector for each sample in the batch.
    Keep in mind that the scheme is a list of Tensorflow symbols, which are stacked by optional batchnorm,
    activation, and dropout in between as specified in InputEmbedderParameters.
    """

    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 scheme: EmbedderScheme=None, batchnorm: bool=False, dropout_rate: float=0.0,
                 name: str= "embedder", input_rescaling: float=1.0, input_offset: float=0.0, input_clipping=None,
                 dense_layer=Dense, is_training=False):
        super().__init__(input_size, activation_function, scheme, batchnorm, dropout_rate, name, input_rescaling,
                         input_offset, input_clipping, dense_layer=dense_layer, is_training=is_training)
        self.return_type = InputTensorEmbedding
        assert scheme is not None, "Custom scheme (a list of callables) must be specified for TensorEmbedder"

    @property
    def schemes(self):
        return {}
