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

from typing import Union

import numpy as np

from rl_coach.core_types import ActionType
from rl_coach.filters.action.action_filter import ActionFilter
from rl_coach.spaces import BoxActionSpace


class LinearBoxToBoxMap(ActionFilter):
    """
    A linear mapping of two box action spaces. For example, if the action space of the
    environment consists of continuous actions between 0 and 1, and we want the agent to choose actions between -1 and 1,
    the LinearBoxToBoxMap can be used to map the range -1 and 1 to the range 0 and 1 in a linear way. This means that the
    action -1 will be mapped to 0, the action 1 will be mapped to 1, and the rest of the actions will be linearly mapped
    between those values.
    """
    def __init__(self,
                 input_space_low: Union[None, int, float, np.ndarray],
                 input_space_high: Union[None, int, float, np.ndarray]):
        """
        :param input_space_low: the low values of the desired action space
        :param input_space_high: the high values of the desired action space
        """
        self.input_space_low = input_space_low
        self.input_space_high = input_space_high
        self.rescale = None
        self.offset = None
        super().__init__()

    def validate_output_action_space(self, output_action_space: BoxActionSpace):
        if not isinstance(output_action_space, BoxActionSpace):
            raise ValueError("BoxActionSpace discretization only works with an output space of type BoxActionSpace. "
                             "The given output space is {}".format(output_action_space))

    def get_unfiltered_action_space(self, output_action_space: BoxActionSpace) -> BoxActionSpace:
        self.input_action_space = BoxActionSpace(output_action_space.shape, self.input_space_low, self.input_space_high)
        self.rescale = \
            (output_action_space.high - output_action_space.low) / (self.input_space_high - self.input_space_low)
        self.offset = output_action_space.low - self.input_space_low
        self.output_action_space = output_action_space
        return self.input_action_space

    def filter(self, action: ActionType) -> ActionType:
        return self.output_action_space.low + (action - self.input_space_low) * self.rescale

