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

from rl_coach.architectures.tensorflow_components.layers import batchnorm_activation_dropout, Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import Embedding
from rl_coach.spaces import SpacesDefinition, BoxActionSpace


class WolpertingerActorHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='tanh',
                 batchnorm: bool=True, dense_layer=Dense, is_training=False):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer, is_training=is_training)
        self.name = 'wolpertinger_actor_head'
        self.return_type = Embedding
        self.action_embedding_width = agent_parameters.algorithm.action_embedding_width
        self.batchnorm = batchnorm
        self.output_scale = self.spaces.action.filtered_action_space.max_abs_range if \
            (hasattr(self.spaces.action, 'filtered_action_space') and
             isinstance(self.spaces.action.filtered_action_space, BoxActionSpace)) \
            else None

    def _build_module(self, input_layer):
        # mean
        pre_activation_policy_value = self.dense_layer(self.action_embedding_width)(input_layer,
                                                                                    name='actor_action_embedding')
        self.proto_action = batchnorm_activation_dropout(input_layer=pre_activation_policy_value,
                                                         batchnorm=self.batchnorm,
                                                         activation_function=self.activation_function,
                                                         dropout_rate=0,
                                                         is_training=self.is_training,
                                                         name="BatchnormActivationDropout_0")[-1]
        if self.output_scale is not None:
            self.proto_action = tf.multiply(self.proto_action, self.output_scale, name='proto_action')

        self.output = [self.proto_action]

    def __str__(self):
        result = [
            'Dense (num outputs = {})'.format(self.action_embedding_width)
        ]
        return '\n'.join(result)
