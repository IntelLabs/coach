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
    Maps a box action space to a box action space.
    For example,
    - the source action space has actions of shape 1 with values between -42 and -10,
    - the target action space has actions of shape 1 with values between 10 and 32
    The mapping will add an offset of 52 to the incoming actions and then multiply them by 22/32 to scale them to the
    target action space
    The shape of the source and target action spaces is always the same
    """
    def __init__(self,
                 input_space_low: Union[None, int, float, np.ndarray],
                 input_space_high: Union[None, int, float, np.ndarray]):
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

