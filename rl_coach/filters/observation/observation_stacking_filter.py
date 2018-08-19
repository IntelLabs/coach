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

import copy
from collections import deque

import numpy as np

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


class LazyStack(object):
    """
    A lazy version of np.stack which avoids copying the memory until it is
    needed.
    """

    def __init__(self, history, axis=None):
        self.history = copy.copy(history)
        self.axis = axis

    def __array__(self, dtype=None):
        array = np.stack(self.history, axis=self.axis)
        if dtype is not None:
            array = array.astype(dtype)
        return array


class ObservationStackingFilter(ObservationFilter):
    """
    Stack the current state observation on top of several previous observations.
    This filter is stateful since it stores the previous step result and depends on it.
    The filter adds an additional dimension to the output observation.

    Warning!!! The filter replaces the observation with a LazyStack object, so no filters should be
    applied after this filter. applying more filters will cause the LazyStack object to be converted to a numpy array
    and increase the memory footprint.
    """
    def __init__(self, stack_size: int, stacking_axis: int=-1):
        """
        :param stack_size: the number of previous observations in the stack
        :param stacking_axis: the axis on which to stack the observation on
        """
        super().__init__()
        self.stack_size = stack_size
        self.stacking_axis = stacking_axis
        self.stack = []

        if stack_size <= 0:
            raise ValueError("The stack shape must be a positive number")
        if type(stack_size) != int:
            raise ValueError("The stack shape must be of int type")

    @property
    def next_filter(self) -> 'InputFilter':
        return self._next_filter

    @next_filter.setter
    def next_filter(self, val: 'InputFilter'):
        raise ValueError("ObservationStackingFilter can have no other filters after it since they break its "
                         "functionality")

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if len(self.stack) > 0 and not input_observation_space.val_matches_space_definition(self.stack[-1]):
            raise ValueError("The given input observation space is different than the observations already stored in"
                             "the filters memory")
        if input_observation_space.num_dimensions <= self.stacking_axis:
            raise ValueError("The stacking axis is larger than the number of dimensions in the observation space")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:

        if len(self.stack) == 0:
            self.stack = deque([observation] * self.stack_size, maxlen=self.stack_size)
        else:
            if update_internal_state:
                self.stack.append(observation)
        observation = LazyStack(self.stack, self.stacking_axis)

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        if self.stacking_axis == -1:
            input_observation_space.shape = np.append(input_observation_space.shape, values=[self.stack_size], axis=0)
        else:
            input_observation_space.shape = np.insert(input_observation_space.shape, obj=self.stacking_axis,
                                                     values=[self.stack_size], axis=0)
        return input_observation_space

    def reset(self) -> None:
        self.stack = []
