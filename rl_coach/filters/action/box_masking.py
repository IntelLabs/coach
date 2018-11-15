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


class BoxMasking(ActionFilter):
    """
    Masks part of the action space to enforce the agent to work in a defined space. For example,
    if the original action space is between -1 and 1, then this filter can be used in order to constrain the agent actions
    to the range 0 and 1 instead. This essentially masks the range -1 and 0 from the agent.
    The resulting action space will be shifted and will always start from 0 and have the size of the unmasked area.
    """
    def __init__(self,
                 masked_target_space_low: Union[None, int, float, np.ndarray],
                 masked_target_space_high: Union[None, int, float, np.ndarray]):
        """
        :param masked_target_space_low: the lowest values that can be chosen in the target action space
        :param masked_target_space_high: the highest values that can be chosen in the target action space
        """
        self.masked_target_space_low = masked_target_space_low
        self.masked_target_space_high = masked_target_space_high
        self.offset = masked_target_space_low
        super().__init__()

    def set_masking(self, masked_target_space_low: Union[None, int, float, np.ndarray],
                    masked_target_space_high: Union[None, int, float, np.ndarray]):
        self.masked_target_space_low = masked_target_space_low
        self.masked_target_space_high = masked_target_space_high
        self.offset = masked_target_space_low
        if self.output_action_space:
            self.validate_output_action_space(self.output_action_space)
            self.input_action_space = BoxActionSpace(self.output_action_space.shape,
                                                     low=0,
                                                     high=self.masked_target_space_high - self.masked_target_space_low)

    def validate_output_action_space(self, output_action_space: BoxActionSpace):
        if not isinstance(output_action_space, BoxActionSpace):
            raise ValueError("BoxActionSpace discretization only works with an output space of type BoxActionSpace. "
                             "The given output space is {}".format(output_action_space))
        if self.masked_target_space_low is None or self.masked_target_space_high is None:
            raise ValueError("The masking target space size was not set. Please call set_masking.")
        if not (np.all(output_action_space.low <= self.masked_target_space_low)
                and np.all(self.masked_target_space_low <= output_action_space.high)):
            raise ValueError("The low values for masking the action space ({}) are not within the range of the "
                             "target space (low = {}, high = {})"
                             .format(self.masked_target_space_low, output_action_space.low, output_action_space.high))
        if not (np.all(output_action_space.low <= self.masked_target_space_high)
                and np.all(self.masked_target_space_high <= output_action_space.high)):
            raise ValueError("The high values for masking the action space ({}) are not within the range of the "
                             "target space (low = {}, high = {})"
                             .format(self.masked_target_space_high, output_action_space.low, output_action_space.high))

    def get_unfiltered_action_space(self, output_action_space: BoxActionSpace) -> BoxActionSpace:
        self.output_action_space = output_action_space
        self.input_action_space = BoxActionSpace(output_action_space.shape,
                                                 low=0,
                                                 high=self.masked_target_space_high - self.masked_target_space_low)
        return self.input_action_space

    def filter(self, action: ActionType) -> ActionType:
        return action + self.offset
