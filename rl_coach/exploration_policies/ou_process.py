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
from rl_coach.exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters
from rl_coach.spaces import ActionSpace, BoxActionSpace, GoalsSpace


# Based on on the description in:
# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OUProcessParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = 0.01

    @property
    def path(self):
        return 'rl_coach.exploration_policies.ou_process:OUProcess'


# Ornstein-Uhlenbeck process
class OUProcess(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace, mu: float=0, theta: float=0.15, sigma: float=0.2, dt: float=0.01):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)
        self.mu = float(mu) * np.ones(self.action_space.shape)
        self.theta = float(theta)
        self.sigma = float(sigma) * np.ones(self.action_space.shape)
        self.state = np.zeros(self.action_space.shape)
        self.dt = dt

        if not (isinstance(action_space, BoxActionSpace) or isinstance(action_space, GoalsSpace)):
            raise ValueError("OU process exploration works only for continuous controls."
                             "The given action space is of type: {}".format(action_space.__class__.__name__))

    def reset(self):
        self.state = np.zeros(self.action_space.shape)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if self.phase == RunPhase.TRAIN:
            noise = self.noise()
        else:
            noise = np.zeros(self.action_space.shape)

        action = action_values.squeeze() + noise

        return action

    def get_control_param(self):
        if self.phase == RunPhase.TRAIN:
            return self.state
        else:
            return np.zeros(self.action_space.shape)
