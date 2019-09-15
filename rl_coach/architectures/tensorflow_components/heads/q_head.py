# #
# # Copyright (c) 2017 Intel Corporation
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
#
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.losses import Huber
# from tensorflow.keras.losses import MeanSquaredError
#
#
# from rl_coach.architectures.tensorflow_components.heads.head import Head
# from rl_coach.base_parameters import AgentParameters
# from rl_coach.core_types import QActionStateValue
# from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace
#
# from typing import Union
#
# class QHead(Head):
#     # def __init__(self,
#     #              agent_parameters: AgentParameters,
#     #              spaces: SpacesDefinition,
#     #              network_name: str,
#     #              head_idx: int = 0,
#     #              is_local: bool = True,
#     #              activation_function: str = 'relu',
#     #              **kwargs):
#
#     def __init__(self,
#                  agent_parameters: AgentParameters,
#                  spaces: SpacesDefinition,
#                  network_name: str,
#                  head_type_idx: int = 0,
#                  loss_weight: float = 1.,
#                  is_local: bool = True,
#                  activation_function: str = 'relu',
#                  dense_layer: None = None,
#                  #loss_type: Union[HuberLoss, L2Loss] = L2Loss,
#                  loss_type: Union[Huber, MeanSquaredError] = MeanSquaredError,
#                  **kwargs) -> None:
#         """
#          Q-Value Head for predicting state-action Q-Values.
#
#          :param agent_parameters: containing algorithm parameters, but currently unused.
#          :param spaces: containing action spaces used for defining size of network output.
#          :param network_name: name of head network. currently unused.
#          :param head_type_idx: index of head network. currently unused.
#          :param is_local: flag to denote if network is local. currently unused.
#          :param activation_function: activation function to use between layers.
#          """
#         super().__init__(**kwargs)
#         #self.name = 'q_values_head'
#         self.spaces = spaces
#         if isinstance(self.spaces.action, BoxActionSpace):
#             self.num_actions = 1
#         elif isinstance(self.spaces.action, DiscreteActionSpace):
#             self.num_actions = len(self.spaces.action.actions)
#         else:
#             raise ValueError(
#                 'QHead does not support action spaces of type: {class_name}'.format(
#                     class_name=self.spaces.action.__class__.__name__,))
#         self.return_type = QActionStateValue
#         self.q_head = keras.layers.Dense(self.num_actions, activation=keras.activations.get(activation_function))
#
#     def call(self, inputs, **kwargs):
#         q_value = self.q_head(inputs)
#         #
#
#
#         return q_value
#
#     def __str__(self):
#         result = [
#             "Dense (num outputs = {})".format(self.num_actions)
#         ]
#         return '\n'.join(result)



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
from typing import Union
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace


class QHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int = 0,
                 loss_weight: float = 1.,
                 is_local: bool = True,
                 activation_function: str = 'relu',
                 dense_layer: None = None) -> None:
        """
        Q-Value Head for predicting state-action Q-Values.

        :param agent_parameters: containing algorithm parameters, but currently unused.
        :param spaces: containing action spaces used for defining size of network output.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        :param loss_type: loss function to use.
        """
        super(QHead, self).__init__(agent_parameters, spaces, network_name, head_type_idx, loss_weight,
                                    is_local, activation_function, dense_layer)
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = 1
        elif isinstance(self.spaces.action, DiscreteActionSpace):
            self.num_actions = len(self.spaces.action.actions)
        self.return_type = QActionStateValue

        self.dense = keras.layers.Dense(units=self.num_actions)

    def call(self, inputs, **kwargs):
        """
        Used for forward pass through Q-Value head network.

        :param inputs: middleware state representation, of shape (batch_size, in_channels).
        :return: predicted state-action q-values, of shape (batch_size, num_actions).
        """
        q_value = self.dense(inputs)
        return q_value
