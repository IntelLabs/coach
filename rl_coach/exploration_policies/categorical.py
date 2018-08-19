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
from rl_coach.spaces import ActionSpace


class CategoricalParameters(ExplorationParameters):
    @property
    def path(self):
        return 'rl_coach.exploration_policies.categorical:Categorical'


class Categorical(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if self.phase == RunPhase.TRAIN:
            # choose actions according to the probabilities
            return np.random.choice(self.action_space.actions, p=action_values)
        else:
            # take the action with the highest probability
            return np.argmax(action_values)

    def get_control_param(self):
        return 0
