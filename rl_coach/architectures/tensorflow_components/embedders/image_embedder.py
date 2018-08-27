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

from rl_coach.architectures.tensorflow_components.architecture import Conv2d, Dense
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedder
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.core_types import InputImageEmbedding


class ImageEmbedder(InputEmbedder):
    """
    An input embedder that performs convolutions on the input and then flattens the result.
    The embedder is intended for image like inputs, where the channels are expected to be the last axis.
    The embedder also allows custom rescaling of the input prior to the neural network.
    """

    def __init__(self, input_size: List[int], activation_function=tf.nn.relu,
                 scheme: EmbedderScheme=EmbedderScheme.Medium, batchnorm: bool=False, dropout: bool=False,
                 name: str= "embedder", input_rescaling: float=255.0, input_offset: float=0.0, input_clipping=None,
                 dense_layer=Dense):
        super().__init__(input_size, activation_function, scheme, batchnorm, dropout, name, input_rescaling,
                         input_offset, input_clipping, dense_layer=dense_layer)
        self.return_type = InputImageEmbedding
        if len(input_size) != 3 and scheme != EmbedderScheme.Empty:
            raise ValueError("Image embedders expect the input size to have 3 dimensions. The given size is: {}"
                             .format(input_size))

    @property
    def schemes(self):
        return {
            EmbedderScheme.Empty:
                [],

            EmbedderScheme.Shallow:
                [
                    Conv2d([32, 3, 1])
                ],

            # atari dqn
            EmbedderScheme.Medium:
                [
                    Conv2d([32, 8, 4]),
                    Conv2d([64, 4, 2]),
                    Conv2d([64, 3, 1])
                ],

            # carla
            EmbedderScheme.Deep: \
                [
                    Conv2d([32, 5, 2]),
                    Conv2d([32, 3, 1]),
                    Conv2d([64, 3, 2]),
                    Conv2d([64, 3, 1]),
                    Conv2d([128, 3, 2]),
                    Conv2d([128, 3, 1]),
                    Conv2d([256, 3, 2]),
                    Conv2d([256, 3, 1])
                ]
        }


