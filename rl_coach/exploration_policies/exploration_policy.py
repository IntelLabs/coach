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

from rl_coach.base_parameters import Parameters
from rl_coach.core_types import RunPhase, ActionType
from rl_coach.spaces import ActionSpace


class ExplorationParameters(Parameters):
    def __init__(self):
        self.action_space = None

    @property
    def path(self):
        return 'rl_coach.exploration_policies.exploration_policy:ExplorationPolicy'


class ExplorationPolicy(object):
    """
    An exploration policy takes the predicted actions or action values from the agent, and selects the action to
    actually apply to the environment using some predefined algorithm.
    """
    def __init__(self, action_space: ActionSpace):
        """
        :param action_space: the action space used by the environment
        """
        self.phase = RunPhase.HEATUP
        self.action_space = action_space

    def reset(self):
        """
        Used for resetting the exploration policy parameters when needed
        :return: None
        """
        pass

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        """
        Given a list of values corresponding to each action, 
        choose one actions according to the exploration policy
        :param action_values: A list of action values
        :return: The chosen action
        """
        if self.__class__ == ExplorationPolicy:
            raise ValueError("The ExplorationPolicy class is an abstract class and should not be used directly. "
                             "Please set the exploration parameters to point to an inheriting class like EGreedy or "
                             "AdditiveNoise")
        else:
            raise ValueError("The get_action function should be overridden in the inheriting exploration class")

    def change_phase(self, phase):
        """
        Change between running phases of the algorithm
        :param phase: Either Heatup or Train
        :return: none
        """
        self.phase = phase

    def requires_action_values(self) -> bool:
        """
        Allows exploration policies to define if they require the action values for the current step.
        This can save up a lot of computation. For example in e-greedy, if the random value generated is smaller
        than epsilon, the action is completely random, and the action values don't need to be calculated
        :return: True if the action values are required. False otherwise
        """
        return True

    def get_control_param(self):
        return 0
