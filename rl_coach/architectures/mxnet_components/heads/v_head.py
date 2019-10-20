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

import mxnet as mx
from mxnet.gluon.loss import Loss, HuberLoss, L2Loss
from mxnet.gluon import nn
from rl_coach.architectures.mxnet_components.heads.head import Head, HeadLoss, LossInputSchema,\
    NormalizedRSSInitializer
from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import VStateValue
from rl_coach.spaces import SpacesDefinition

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class VHeadLoss(HeadLoss):
    def __init__(self, loss_type: Loss=L2Loss, weight: float=1, batch_axis: int=0) -> None:
        """
        Loss for Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super(VHeadLoss, self).__init__(weight=weight, batch_axis=batch_axis)
        with self.name_scope():
            self.loss_fn = loss_type(weight=weight, batch_axis=batch_axis)

    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            head_outputs=['pred'],
            agent_inputs=[],
            targets=['target']
        )

    def loss_forward(self,
                     F: ModuleType,
                     pred: nd_sym_type,
                     target: nd_sym_type) -> List[Tuple[nd_sym_type, str]]:
        """
        Used for forward pass through loss computations.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param pred: state values predicted by VHead network, of shape (batch_size).
        :param target: actual state values, of shape (batch_size).
        :return: loss, of shape (batch_size).
        """
        loss = self.loss_fn(pred, target).mean()
        return [(loss, LOSS_OUT_TYPE_LOSS)]


class VHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='relu',
                 dense_layer: None=None,
                 loss_type: Union[HuberLoss, L2Loss]=L2Loss):
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
        :param loss_type: loss function with default of mean squared error (i.e. L2Loss), or alternatively HuberLoss.
        """
        super(VHead, self).__init__(agent_parameters, spaces, network_name, head_type_idx, loss_weight,
                                    is_local, activation_function, dense_layer)
        assert (loss_type == L2Loss) or (loss_type == HuberLoss), "Only expecting L2Loss or HuberLoss."
        self.loss_type = loss_type
        self.return_type = VStateValue
        with self.name_scope():
            self.dense = nn.Dense(units=1, weight_initializer=NormalizedRSSInitializer(1.0))

    def loss(self) -> Loss:
        """
        Specifies loss block to be used for specific value head implementation.

        :return: loss block (can be called as function) for outputs returned by the head network.
        """
        return VHeadLoss(loss_type=self.loss_type, weight=self.loss_weight)

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type) -> nd_sym_type:
        """
        Used for forward pass through value head network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: middleware state representation, of shape (batch_size, in_channels).
        :return: final output of value network, of shape (batch_size).
        """
        return self.dense(x).squeeze(axis=1)
