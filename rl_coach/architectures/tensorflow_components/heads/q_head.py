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


from typing import Union, List, Tuple
from types import ModuleType

from tensorflow.keras.losses import Loss, Huber, MeanSquaredError
from rl_coach.architectures.tensorflow_components.heads.head import Head, HeadLoss, LossInputSchema
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace

from tensorflow import keras

LOSS_OUT_TYPE_LOSS = 'loss'
LOSS_OUT_TYPE_REGULARIZATION = 'regularization'


class QHeadLoss(HeadLoss):
    def __init__(self, loss_type: Loss = MeanSquaredError, **kwargs):
        """
        Loss for Q-Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super().__init__(**kwargs)

        self.loss_fn = keras.losses.get(loss_type)


    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            head_outputs=['pred'],
            agent_inputs=[],
            targets=['target']
        )


    def call(self, target, pred):
        """
        Used for forward pass through loss computations.
        :param pred: state-action q-values predicted by QHead network, of shape (batch_size, num_actions).
        :param target: actual state-action q-values, of shape (batch_size, num_actions).
        :return: loss, of shape (batch_size).
        """
        # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
        #  This way, Keras can apply class weights or sample weights when requested.
        loss = self.loss_fn(pred, target).mean()
        return [(loss, LOSS_OUT_TYPE_LOSS)]





class QHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='relu',
                 dense_layer: None=None,
                 loss_type: Union[Huber, MeanSquaredError]=MeanSquaredError) -> None:
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
        assert (loss_type == MeanSquaredError) or (loss_type == HuberLoss), "Only expecting L2Loss or HuberLoss."
        self.loss_type = loss_type

        self.dense = keras.layers.Dense(units=self.num_actions)

    def loss(self) -> Loss:
        """
        Specifies loss block to be used for specific value head implementation.

        :return: loss block (can be called as function) for outputs returned by the head network.
        """
        return QHeadLoss(loss_type=self.loss_type, weight=self.loss_weight)

    def call(self, inputs, **kwargs):
        """
        Used for forward pass through Q-Value head network.

        :param x: middleware state representation, of shape (batch_size, in_channels).
        :return: predicted state-action q-values, of shape (batch_size, num_actions).
        """
        q_value = self.dense(inputs)
        return q_value
