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

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


class ObservationRGBToYFilter(ObservationFilter):
    """
    Converts the observation in the current state to gray scale (Y channel).
    The channels axis is assumed to be the last axis
    """
    def __init__(self):
        super().__init__()

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if input_observation_space.num_dimensions != 3:
            raise ValueError("The rescale filter only applies to image observations where the number of dimensions is"
                             "3 (RGB). The number of dimensions defined for the input observation was {}"
                             .format(input_observation_space.num_dimensions))
        if input_observation_space.shape[-1] != 3:
            raise ValueError("The observation space is expected to have 3 channels in the 1st dimension. The number of "
                             "dimensions received is {}".format(input_observation_space.shape[-1]))

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:

        # rgb to y
        r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        observation = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape = input_observation_space.shape[:-1]
        return input_observation_space
