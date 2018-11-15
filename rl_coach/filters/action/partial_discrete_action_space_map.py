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

from rl_coach.core_types import ActionType
from rl_coach.filters.action.action_filter import ActionFilter
from rl_coach.spaces import DiscreteActionSpace, ActionSpace


class PartialDiscreteActionSpaceMap(ActionFilter):
    """
    Partial map of two countable action spaces. For example, consider an environment
    with a MultiSelect action space (select multiple actions at the same time, such as jump and go right), with 8 actual
    MultiSelect actions. If we want the agent to be able to select only 5 of those actions by their index (0-4), we can
    map a discrete action space with 5 actions into the 5 selected MultiSelect actions. This will both allow the agent to
    use regular discrete actions, and mask 3 of the actions from the agent.
    """
    def __init__(self, target_actions: List[ActionType]=None, descriptions: List[str]=None):
        """
        :param target_actions: A partial list of actions from the target space to map to.
        :param descriptions: a list of descriptions of each of the actions
        """
        self.target_actions = target_actions
        self.descriptions = descriptions
        super().__init__()

    def validate_output_action_space(self, output_action_space: ActionSpace):
        if not self.target_actions:
            raise ValueError("The target actions were not set")
        for v in self.target_actions:
            if not output_action_space.val_matches_space_definition(v):
                raise ValueError("The values in the output actions ({}) do not match the output action "
                                 "space definition ({})".format(v, output_action_space))

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> DiscreteActionSpace:
        self.output_action_space = output_action_space
        self.input_action_space = DiscreteActionSpace(len(self.target_actions), self.descriptions)
        return self.input_action_space

    def filter(self, action: ActionType) -> ActionType:
        return self.target_actions[action]

    def reverse_filter(self, action: ActionType) -> ActionType:
        return [(action == x).all() for x in self.target_actions].index(True)

