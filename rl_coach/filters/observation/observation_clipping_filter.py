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


class ObservationClippingFilter(ObservationFilter):
    """
    Clip the observation values using the given ranges
    """
    def __init__(self, clipping_low: float=-np.inf, clipping_high: float=np.inf):
        """
        :param clipping_low: The minimum value to allow after normalizing the observation
        :param clipping_high: The maximum value to allow after normalizing the observation
        """
        super().__init__()
        self.clip_min = clipping_low
        self.clip_max = clipping_high

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        observation = np.clip(observation, self.clip_min, self.clip_max)

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        return input_observation_space
