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

import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.exploration_policies.continuous_entropy import ContinuousEntropyParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, CompoundActionSpace
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps, indent_string


class AiWeekHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='tanh',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'workshop_head'
        self.return_type = ActionProbabilities
        self.exploration_policy = agent_parameters.exploration


    def _build_module(self, input_layer):
        self.actions = []
        self.input = self.actions
        self.policy_distributions = []

        num_actions = len(self.spaces.action.actions)

        self._build_net_head(input_layer, num_actions)

        # calculate loss
        action_log_prob = tf.add_n([dist.log_prob(action) for dist, action in zip(self.policy_distributions, self.actions)])
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.target = self.advantages

        self.loss = -tf.reduce_mean(action_log_prob * self.advantages)
        tf.losses.add_loss(self.loss)

    def _build_net_head(self, input_layer, num_actions):

        self.actions.append(tf.placeholder(tf.int32, [None], name="actions"))

        p_right = self.dense_layer(num_actions)(input_layer, name='fc')
        self.policy_probs = tf.nn.softmax(p_right, name="policy")

        # define the distributions for the policy and the old policy
        # (the + eps is to prevent probability 0 which will cause the log later on to be -inf)
        policy_distribution = tf.contrib.distributions.Categorical(probs=(self.policy_probs + eps))
        self.policy_distributions.append(policy_distribution)
        self.output.append(self.policy_probs)

