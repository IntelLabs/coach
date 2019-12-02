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

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Lambda

from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import VStateValue
from rl_coach.spaces import SpacesDefinition


class VHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='relu',
                 dense_layer: None=None):
        """
        Value Head for predicting state values.
        :param agent_parameters: containing algorithm parameters, but currently unused.
        :param spaces: containing action spaces, but currently unused.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super(VHead, self).__init__(agent_parameters,
                                    spaces,
                                    network_name,
                                    head_type_idx,
                                    loss_weight,
                                    is_local,
                                    activation_function,
                                    dense_layer)

        self.return_type = VStateValue
        self.dense = keras.layers.Dense(units=1)

    def call(self, inputs):
        """
        Used for forward pass through Q-Value head network.

        :param inputs: middleware state representation, of shape (batch_size, in_channels).
        :return: predicted state-action q-values, of shape (batch_size, num_actions).
        """
        value = self.dense(inputs)
        return value


def value_head(input_dim, output_dim):
    inputs = Input(shape=([input_dim]))
    value = Dense(units=output_dim, name="value_output")(inputs)
    model = keras.Model(name='value_head', inputs=inputs, outputs=value)
    return model
