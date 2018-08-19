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


class ObservationToUInt8Filter(ObservationFilter):
    """
    Converts the observation values to be uint8 values between 0 and 255.
    It first scales the observation values to fit in the range and then converts them to uint8.
    """
    def __init__(self, input_low: float, input_high: float):
        super().__init__()
        self.input_low = input_low
        self.input_high = input_high

        if input_high <= input_low:
            raise ValueError("The input observation space high values can be less or equal to the input observation "
                             "space low values")

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if np.all(input_observation_space.low != self.input_low) or \
                np.all(input_observation_space.high != self.input_high):
            raise ValueError("The observation space values range don't match the configuration of the filter."
                             "The configuration is: low = {}, high = {}. The actual values are: low = {}, high = {}"
                             .format(self.input_low, self.input_high,
                                     input_observation_space.low, input_observation_space.high))

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        # scale to 0-1
        observation = (observation - self.input_low) / (self.input_high - self.input_low)

        # scale to 0-255
        observation *= 255

        observation = observation.astype('uint8')

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.low = 0
        input_observation_space.high = 255
        return input_observation_space
