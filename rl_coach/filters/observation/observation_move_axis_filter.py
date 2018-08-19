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

import numpy as np

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace, PlanarMapsObservationSpace


class ObservationMoveAxisFilter(ObservationFilter):
    """
    Move an axis of the observation to a different place.
    """
    def __init__(self, axis_origin: int = None, axis_target: int=None):
        super().__init__()
        self.axis_origin = axis_origin
        self.axis_target = axis_target

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        shape = input_observation_space.shape
        if not -len(shape) <= self.axis_origin < len(shape) or not -len(shape) <= self.axis_target < len(shape):
            raise ValueError("The given axis does not exist in the context of the input observation shape. ")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        return np.moveaxis(observation, self.axis_origin, self.axis_target)

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        axis_size = input_observation_space.shape[self.axis_origin]
        input_observation_space.shape = np.delete(input_observation_space.shape, self.axis_origin)
        if self.axis_target == -1:
            input_observation_space.shape = np.append(input_observation_space.shape, axis_size)
        elif self.axis_target < -1:
            input_observation_space.shape = np.insert(input_observation_space.shape, self.axis_target+1, axis_size)
        else:
            input_observation_space.shape = np.insert(input_observation_space.shape, self.axis_target, axis_size)

        # move the channels axis according to the axis change
        if isinstance(input_observation_space, PlanarMapsObservationSpace):
            if input_observation_space.channels_axis == self.axis_origin:
                input_observation_space.channels_axis = self.axis_target
            elif input_observation_space.channels_axis == self.axis_target:
                input_observation_space.channels_axis = self.axis_origin
            elif self.axis_origin < input_observation_space.channels_axis < self.axis_target:
                input_observation_space.channels_axis -= 1
            elif self.axis_target < input_observation_space.channels_axis < self.axis_origin:
                input_observation_space.channels_axis += 1

        return input_observation_space
