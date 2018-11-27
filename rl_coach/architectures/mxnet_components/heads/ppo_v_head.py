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


from typing import List, Tuple, Union
from types import ModuleType

import mxnet as mx
from mxnet.gluon import nn
from rl_coach.architectures.mxnet_components.heads.head import Head, HeadLoss, LossInputSchema,\
    NormalizedRSSInitializer
from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class PPOVHeadLoss(HeadLoss):
    def __init__(self, clip_likelihood_ratio_using_epsilon: float, weight: float=1, batch_axis: int=0) -> None:
        """
        Loss for PPO Value network.
        Schulman implemented this extension in OpenAI baselines for PPO2
        See https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py#L72

        :param clip_likelihood_ratio_using_epsilon: epsilon to use for likelihood ratio clipping.
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super(PPOVHeadLoss, self).__init__(weight=weight, batch_axis=batch_axis)
        self.weight = weight
        self.clip_likelihood_ratio_using_epsilon = clip_likelihood_ratio_using_epsilon

    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            head_outputs=['new_policy_values'],
            agent_inputs=['old_policy_values'],
            targets=['target_values']
        )

    def loss_forward(self,
                     F: ModuleType,
                     new_policy_values: nd_sym_type,
                     old_policy_values: nd_sym_type,
                     target_values: nd_sym_type) -> List[Tuple[nd_sym_type, str]]:
        """
        Used for forward pass through loss computations.
        Calculates two losses (L2 and a clipped difference L2 loss) and takes the maximum of the two.
        Works with batches of data, and optionally time_steps, but be consistent in usage: i.e. if using time_step,
        new_policy_values, old_policy_values and target_values all must include a time_step dimension.

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param new_policy_values: values predicted by PPOVHead network,
            of shape (batch_size) or
            of shape (batch_size, time_step).
        :param old_policy_values: values predicted by old value network,
            of shape (batch_size) or
            of shape (batch_size, time_step).
        :param target_values: actual state values,
            of shape (batch_size) or
            of shape (batch_size, time_step).
        :return: loss, of shape (batch_size).
        """
        # L2 loss
        value_loss_1 = (new_policy_values - target_values).square()
        # Clipped difference L2 loss
        diff = new_policy_values - old_policy_values
        clipped_diff = diff.clip(a_min=-self.clip_likelihood_ratio_using_epsilon,
                                 a_max=self.clip_likelihood_ratio_using_epsilon)
        value_loss_2 = (old_policy_values + clipped_diff - target_values).square()
        # Maximum of the two losses, element-wise maximum.
        value_loss_max = mx.nd.stack(value_loss_1, value_loss_2).max(axis=0)
        # Aggregate over temporal axis, adding if doesn't exist (hense reshape)
        value_loss_max_w_time = value_loss_max.reshape(shape=(0, -1))
        value_loss = value_loss_max_w_time.mean(axis=1)
        # Weight the loss (and average over samples of batch)
        value_loss_weighted = value_loss.mean() * self.weight
        return [(value_loss_weighted, LOSS_OUT_TYPE_LOSS)]


class PPOVHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool = True,
                 activation_function: str='relu',
                 dense_layer: None=None) -> None:
        """
        PPO Value Head for predicting state values.

        :param agent_parameters: containing algorithm parameters, but currently unused.
        :param spaces: containing action spaces, but currently unused.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super(PPOVHead, self).__init__(agent_parameters, spaces, network_name, head_type_idx, loss_weight, is_local,
                                       activation_function, dense_layer=dense_layer)
        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.return_type = ActionProbabilities
        with self.name_scope():
            self.dense = nn.Dense(units=1, weight_initializer=NormalizedRSSInitializer(1.0))

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type) -> nd_sym_type:
        """
        Used for forward pass through value head network.

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param x: middleware state representation, of shape (batch_size, in_channels).
        :return: final value output of network, of shape (batch_size).
        """
        return self.dense(x).squeeze(axis=1)

    def loss(self) -> mx.gluon.loss.Loss:
        """
        Specifies loss block to be used for specific value head implementation.

        :return: loss block (can be called as function) for outputs returned by the value head network.
        """
        return PPOVHeadLoss(self.clip_likelihood_ratio_using_epsilon, weight=self.loss_weight)
