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

from rl_coach.core_types import RunPhase, ActionType, EnvironmentSteps
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.e_greedy import EGreedy, EGreedyParameters
from rl_coach.exploration_policies.exploration_policy import ExplorationParameters
from rl_coach.schedules import Schedule, LinearSchedule, PieceWiseSchedule
from rl_coach.spaces import ActionSpace


class UCBParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.architecture_num_q_heads = 10
        self.bootstrapped_data_sharing_probability = 1.0
        self.epsilon_schedule = PieceWiseSchedule([
            (LinearSchedule(1, 0.1, 1000000), EnvironmentSteps(1000000)),
            (LinearSchedule(0.1, 0.01, 4000000), EnvironmentSteps(4000000))
        ])
        self.lamb = 0.1

    @property
    def path(self):
        return 'rl_coach.exploration_policies.ucb:UCB'


class UCB(EGreedy):
    """
    UCB exploration policy is following the upper confidence bound heuristic to sample actions in discrete action spaces.
    It assumes that there are multiple network heads that are predicting action values, and that the standard deviation
    between the heads predictions represents the uncertainty of the agent in each of the actions.
    It then updates the action value estimates to by mean(actions)+lambda*stdev(actions), where lambda is
    given by the user. This exploration policy aims to take advantage of the uncertainty of the agent in its predictions,
    and select the action according to the tradeoff between how uncertain the agent is, and how large it predicts
    the outcome from those actions to be.
    """
    def __init__(self, action_space: ActionSpace, epsilon_schedule: Schedule, evaluation_epsilon: float,
                 architecture_num_q_heads: int, lamb: int,
                 continuous_exploration_policy_parameters: ExplorationParameters = AdditiveNoiseParameters()):
        """
        :param action_space: the action space used by the environment
        :param epsilon_schedule: a schedule for the epsilon values
        :param evaluation_epsilon: the epsilon value to use for evaluation phases
        :param architecture_num_q_heads: the number of q heads to select from
        :param lamb: lambda coefficient for taking the standard deviation into account
        :param continuous_exploration_policy_parameters: the parameters of the continuous exploration policy to use
                                                         if the e-greedy is used for a continuous policy
        """
        super().__init__(action_space, epsilon_schedule, evaluation_epsilon, continuous_exploration_policy_parameters)
        self.num_heads = architecture_num_q_heads
        self.lamb = lamb
        self.std = 0
        self.last_action_values = 0

    def select_head(self):
        pass

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        # action values are none in case the exploration policy is going to select a random action
        if action_values is not None:
            if self.requires_action_values():
                mean = np.mean(action_values, axis=0)
                if self.phase == RunPhase.TRAIN:
                    self.std = np.std(action_values, axis=0)
                    self.last_action_values = mean + self.lamb * self.std
                else:
                    self.last_action_values = mean
        return super().get_action(self.last_action_values)

    def get_control_param(self):
        if self.phase == RunPhase.TRAIN:
            return np.mean(self.std)
        else:
            return 0
