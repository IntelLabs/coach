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
from rl_coach.schedules import Schedule
from rl_coach.spaces import ActionSpace


class BoltzmannParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.temperature_schedule = None

    @property
    def path(self):
        return 'rl_coach.exploration_policies.boltzmann:Boltzmann'



class Boltzmann(ExplorationPolicy):
    """
    The Boltzmann exploration policy is intended for discrete action spaces. It assumes that each of the possible
    actions has some value assigned to it (such as the Q value), and uses a softmax function to convert these values
    into a distribution over the actions. It then samples the action for playing out of the calculated distribution.
    An additional temperature schedule can be given by the user, and will control the steepness of the softmax function.
    """
    def __init__(self, action_space: ActionSpace, temperature_schedule: Schedule):
        """
        :param action_space: the action space used by the environment
        :param temperature_schedule: the schedule for the temperature parameter of the softmax
        """
        super().__init__(action_space)
        self.temperature_schedule = temperature_schedule

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if self.phase == RunPhase.TRAIN:
            self.temperature_schedule.step()
        # softmax calculation
        exp_probabilities = np.exp(action_values / self.temperature_schedule.current_value)
        probabilities = exp_probabilities / np.sum(exp_probabilities)
        # make sure probs sum to 1
        probabilities[-1] = 1 - np.sum(probabilities[:-1])
        # choose actions according to the probabilities
        return np.random.choice(range(self.action_space.shape), p=probabilities)

    def get_control_param(self):
        return self.temperature_schedule.current_value
