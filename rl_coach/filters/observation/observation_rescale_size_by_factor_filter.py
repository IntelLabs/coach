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

from skimage.transform import resize


from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


class ObservationRescaleSizeByFactorFilter(ObservationFilter):
    """
    Rescales an image observation by some factor. For example, the image size
    can be reduced by a factor of 2.
    """
    def __init__(self, rescale_factor: float):
        """
        :param rescale_factor: the factor by which the observation will be rescaled
        """
        super().__init__()
        self.rescale_factor = float(rescale_factor)
        # TODO: allow selecting the channels dim

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if not 2 <= input_observation_space.num_dimensions <= 3:
            raise ValueError("The rescale filter only applies to image observations where the number of dimensions is"
                             "either 2 (grayscale) or 3 (RGB). The number of dimensions defined for the "
                             "output observation was {}".format(input_observation_space.num_dimensions))
        if input_observation_space.num_dimensions == 3 and input_observation_space.shape[-1] != 3:
            raise ValueError("Observations with 3 dimensions must have 3 channels in the last axis (RGB)")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        observation = observation.astype('uint8')
        rescaled_output_size = tuple([int(self.rescale_factor * dim) for dim in observation.shape[:2]])

        if len(observation.shape) == 3:
            rescaled_output_size += (3,)

        # rescale
        observation = resize(observation, rescaled_output_size, anti_aliasing=False, preserve_range=True).astype('uint8')

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape[:2] = (input_observation_space.shape[:2] * self.rescale_factor).astype('int')
        return input_observation_space
