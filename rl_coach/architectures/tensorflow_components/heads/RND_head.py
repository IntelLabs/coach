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
import numpy as np

from rl_coach.architectures.tensorflow_components.layers import Conv2d, BatchnormActivationDropout
from rl_coach.architectures.tensorflow_components.heads.head import Head, Orthogonal
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import Embedding
from rl_coach.spaces import SpacesDefinition


class RNDHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, is_local: bool = True, is_predictor: bool = False):
        super().__init__(agent_parameters, spaces, network_name, head_idx, is_local)
        self.name = 'rnd_head'
        self.return_type = Embedding
        self.is_predictor = is_predictor
        self.activation_function = tf.nn.leaky_relu

        self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        weight_init = Orthogonal(gain=np.sqrt(2))
        input_layer = Conv2d(num_filters=32, kernel_size=8, strides=4)(input_layer, kernel_initializer=weight_init)
        input_layer = BatchnormActivationDropout(activation_function=self.activation_function)(input_layer)[-1]
        input_layer = Conv2d(num_filters=64, kernel_size=4, strides=2)(input_layer, kernel_initializer=weight_init)
        input_layer = BatchnormActivationDropout(activation_function=self.activation_function)(input_layer)[-1]
        input_layer = Conv2d(num_filters=64, kernel_size=3, strides=1)(input_layer, kernel_initializer=weight_init)
        input_layer = BatchnormActivationDropout(activation_function=self.activation_function)(input_layer)[-1]
        input_layer = tf.contrib.layers.flatten(input_layer)

        if self.is_predictor:
            input_layer = self.dense_layer(512)(input_layer, kernel_initializer=weight_init)
            input_layer = BatchnormActivationDropout(activation_function=tf.nn.relu)(input_layer)[-1]
            input_layer = self.dense_layer(512)(input_layer, kernel_initializer=weight_init)
            input_layer = BatchnormActivationDropout(activation_function=tf.nn.relu)(input_layer)[-1]

        self.output = self.dense_layer(512)(input_layer, name='output', kernel_initializer=weight_init)
