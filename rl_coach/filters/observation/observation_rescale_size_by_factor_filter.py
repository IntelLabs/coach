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

from enum import Enum

import scipy.ndimage

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


# imresize interpolation types as defined by scipy here:
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.imresize.html
class RescaleInterpolationType(Enum):
    NEAREST = 'nearest'
    LANCZOS = 'lanczos'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    CUBIC = 'cubic'


class ObservationRescaleSizeByFactorFilter(ObservationFilter):
    """
    Scales the current state observation size by a given factor
    Warning: this requires the input observation to be of type uint8 due to scipy requirements!
    """
    def __init__(self, rescale_factor: float, rescaling_interpolation_type: RescaleInterpolationType):
        """
        :param rescale_factor: the factor by which the observation will be rescaled
        :param rescaling_interpolation_type: the interpolation type for rescaling
        """
        super().__init__()
        self.rescale_factor = float(rescale_factor)  # scipy requires float scale factors
        self.rescaling_interpolation_type = rescaling_interpolation_type
        # TODO: allow selecting the channels dim

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if not 2 <= input_observation_space.num_dimensions <= 3:
            raise ValueError("The rescale filter only applies to image observations where the number of dimensions is"
                             "either 2 (grayscale) or 3 (RGB). The number of dimensions defined for the "
                             "output observation was {}".format(input_observation_space.num_dimensions))
        if input_observation_space.num_dimensions == 3 and input_observation_space.shape[-1] != 3:
            raise ValueError("Observations with 3 dimensions must have 3 channels in the last axis (RGB)")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        # scipy works only with uint8
        observation = observation.astype('uint8')

        # rescale
        observation = scipy.misc.imresize(observation,
                                          self.rescale_factor,
                                          interp=self.rescaling_interpolation_type.value)

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape[:2] = (input_observation_space.shape[:2] * self.rescale_factor).astype('int')
        return input_observation_space
