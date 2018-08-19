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

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.heads.head import Head, HeadParameters
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition


class CategoricalQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='categorical_q_head_params'):
        super().__init__(parameterized_class=CategoricalQHead, activation_function=activation_function, name=name)


class CategoricalQHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str ='relu'):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function)
        self.name = 'categorical_dqn_head'
        self.num_actions = len(self.spaces.action.actions)
        self.num_atoms = agent_parameters.algorithm.atoms
        self.return_type = QActionStateValue

    def _build_module(self, input_layer):
        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.input = [self.actions]

        values_distribution = tf.layers.dense(input_layer, self.num_actions * self.num_atoms, name='output')
        values_distribution = tf.reshape(values_distribution, (tf.shape(values_distribution)[0], self.num_actions,
                                                               self.num_atoms))
        # softmax on atoms dimension
        self.output = tf.nn.softmax(values_distribution)

        # calculate cross entropy loss
        self.distributions = tf.placeholder(tf.float32, shape=(None, self.num_actions, self.num_atoms),
                                            name="distributions")
        self.target = self.distributions
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=values_distribution)
        tf.losses.add_loss(self.loss)
