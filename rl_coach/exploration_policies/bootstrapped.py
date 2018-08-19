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

from typing import List

import numpy as np

from rl_coach.core_types import RunPhase, ActionType
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.e_greedy import EGreedy, EGreedyParameters
from rl_coach.exploration_policies.exploration_policy import ExplorationParameters
from rl_coach.schedules import Schedule, LinearSchedule
from rl_coach.spaces import ActionSpace


class BootstrappedParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.architecture_num_q_heads = 10
        self.bootstrapped_data_sharing_probability = 1.0
        self.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)

    @property
    def path(self):
        return 'rl_coach.exploration_policies.bootstrapped:Bootstrapped'


class Bootstrapped(EGreedy):
    def __init__(self, action_space: ActionSpace, epsilon_schedule: Schedule, evaluation_epsilon: float,
                 architecture_num_q_heads: int,
                 continuous_exploration_policy_parameters: ExplorationParameters = AdditiveNoiseParameters(),):
        """
        :param action_space: the action space used by the environment
        :param epsilon_schedule: a schedule for the epsilon values
        :param evaluation_epsilon: the epsilon value to use for evaluation phases
        :param continuous_exploration_policy_parameters: the parameters of the continuous exploration policy to use
                                                         if the e-greedy is used for a continuous policy
        :param architecture_num_q_heads: the number of q heads to select from
        """
        super().__init__(action_space, epsilon_schedule, evaluation_epsilon, continuous_exploration_policy_parameters)
        self.num_heads = architecture_num_q_heads
        self.selected_head = 0
        self.last_action_values = 0

    def select_head(self):
        self.selected_head = np.random.randint(self.num_heads)

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        # action values are none in case the exploration policy is going to select a random action
        if action_values is not None:
            if self.phase == RunPhase.TRAIN:
                action_values = action_values[self.selected_head]
            else:
                # ensemble voting for evaluation
                top_action_votings = np.argmax(action_values, axis=-1)
                counts = np.bincount(top_action_votings.squeeze())
                top_action = np.argmax(counts)
                # convert the top action to a one hot vector and pass it to e-greedy
                action_values = np.eye(len(self.action_space.actions))[[top_action]]
        self.last_action_values = action_values
        return super().get_action(action_values)

    def get_control_param(self):
        return self.selected_head
