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


import tensorflow as tf
from typing import List, Dict
from rl_coach.architectures.layers import Dense
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedder
from rl_coach.base_parameters import EmbedderScheme
from rl_coach.core_types import InputVectorEmbedding


class VectorEmbedder(InputEmbedder):
    """
    A vector embedder is an input embedder that takes an vector input from the state and produces a vector
    embedding by passing it through a neural network.

    :param params: parameters object containing input_clipping, input_rescaling, batchnorm, activation_function
    and dropout properties.
    """

    def __init__(self, input_size: List[int],
                 activation_function=tf.nn.relu,
                 scheme: EmbedderScheme=EmbedderScheme.Medium,
                 batchnorm: bool=False,
                 dropout_rate: float=0.0,
                 name: str = "embedder",
                 input_rescaling: float=1.0,
                 input_offset: float=0.0,
                 input_clipping=None,
                 is_training=False):

        super().__init__(input_size, activation_function, scheme, batchnorm, dropout_rate, name,
                         input_rescaling, input_offset, input_clipping,
                         is_training=is_training)

        self.return_type = InputVectorEmbedding
        if len(self.input_size) != 1 and scheme != EmbedderScheme.Empty:
            raise ValueError("The input size of a vector embedder must contain only a single dimension")

    @property
    def schemes(self) -> Dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used. Are used
        to create Block when VectorEmbedder is initialised.

        :return: dictionary of schemes, with key of type EmbedderScheme enum and value being list of Tensorflow layers.
        """
        return {
            EmbedderScheme.Empty:
                [],

            EmbedderScheme.Shallow:
                [
                    Dense(128)
                ],

            # Use for DQN
            EmbedderScheme.Medium:
                [
                    Dense(256)
                ],

            # Use for Carla
            EmbedderScheme.Deep: \
                [
                    Dense(128),
                    Dense(128),
                    Dense(128)
                ]
        }
