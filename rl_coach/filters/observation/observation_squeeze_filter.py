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
from rl_coach.spaces import ObservationSpace


class ObservationSqueezeFilter(ObservationFilter):
    """
    Removes redundant axes from the observation, which are axes with a dimension of 1.
    """
    def __init__(self, axis: int = None):
        """
        :param axis: Specifies which axis to remove. If set to None, all the axes of size 1 will be removed.
        """
        super().__init__()
        self.axis = axis

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if self.axis is None:
            return

        shape = input_observation_space.shape
        if self.axis >= len(shape) or self.axis < -len(shape):
            raise ValueError("The given axis does not exist in the context of the input observation shape. ")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        return observation.squeeze(axis=self.axis)

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        dummy_tensor = np.random.rand(*tuple(input_observation_space.shape))
        input_observation_space.shape = dummy_tensor.squeeze(axis=self.axis).shape
        return input_observation_space
