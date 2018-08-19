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

from rl_coach.core_types import ActionType
from rl_coach.filters.filter import Filter
from rl_coach.spaces import ActionSpace


class ActionFilter(Filter):
    def __init__(self, input_action_space: ActionSpace=None):
        self.input_action_space = input_action_space
        self.output_action_space = None
        super().__init__()

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> ActionSpace:
        """
        This function should contain the logic for getting the unfiltered action space
        :param output_action_space: the output action space
        :return: the unfiltered action space
        """
        return output_action_space

    def validate_output_action_space(self, output_action_space: ActionSpace):
        """
        A function that implements validation of the output action space
        :param output_action_space: the input action space
        :return: None
        """
        pass

    def validate_output_action(self, action: ActionType):
        """
        A function that verifies that the given action is in the expected output action space
        :param action: an action to validate
        :return: None
        """
        if not self.output_action_space.val_matches_space_definition(action):
            raise ValueError("The given action ({}) does not match the action space ({})"
                             .format(action, self.output_action_space))

    def filter(self, action: ActionType) -> ActionType:
        """
        A function that transforms from the agent's action space to the environment's action space
        :param action: an action to transform
        :return: transformed action
        """
        raise NotImplementedError("")

    def reverse_filter(self, action: ActionType) -> ActionType:
        """
        A function that transforms from the environment's action space to the agent's action space
        :param action: an action to transform
        :return: transformed action
        """
        raise NotImplementedError("")