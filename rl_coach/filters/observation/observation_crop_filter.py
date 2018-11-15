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
from typing import Union, Tuple

import numpy as np

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


class ObservationCropFilter(ObservationFilter):
    """
    Crops the size of the observation to a given crop window. For example, in Atari, the
    observations are images with a shape of 210x160. Usually, we will want to crop the size of the observation to a
    square of 160x160 before rescaling them.
    """
    def __init__(self, crop_low: np.ndarray=None, crop_high: np.ndarray=None):
        """
        :param crop_low: a vector where each dimension describes the start index for cropping the observation in the
                         corresponding dimension. a negative value of -1 will be mapped to the max size
        :param crop_high: a vector where each dimension describes the end index for cropping the observation in the
                          corresponding dimension. a negative value of -1 will be mapped to the max size
        """
        super().__init__()
        if crop_low is None and crop_high is None:
            raise ValueError("At least one of crop_low and crop_high should be set to a real value. ")
        if crop_low is None:
            crop_low = np.array([0] * len(crop_high))
        if crop_high is None:
            crop_high = np.array([-1] * len(crop_low))

        self.crop_low = crop_low
        self.crop_high = crop_high

        for h, l in zip(crop_high, crop_low):
            if h < l and h != -1:
                raise ValueError("Some of the cropping low values are higher than cropping high values")
        if np.any(crop_high < -1) or np.any(crop_low < -1):
            raise ValueError("Cropping values cannot be negative")
        if crop_low.shape != crop_high.shape:
            raise ValueError("The low values and high values for cropping must have the same number of dimensions")
        if crop_low.dtype != int or crop_high.dtype != int:
            raise ValueError("The crop values should be int values, instead they are defined as: {} and {}"
                             .format(crop_low.dtype, crop_high.dtype))

    def _replace_negative_one_in_crop_size(self, crop_size: np.ndarray, observation_shape: Union[Tuple, np.ndarray]):
        # replace -1 with the max size
        crop_size = crop_size.copy()
        for i in range(len(observation_shape)):
            if crop_size[i] == -1:
                crop_size[i] = observation_shape[i]
        return crop_size

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        crop_high = self._replace_negative_one_in_crop_size(self.crop_high, input_observation_space.shape)
        crop_low = self._replace_negative_one_in_crop_size(self.crop_low, input_observation_space.shape)
        if np.any(crop_high > input_observation_space.shape) or \
                np.any(crop_low > input_observation_space.shape):
            raise ValueError("The cropping values are outside of the observation space")
        if not input_observation_space.is_point_in_space_shape(crop_low) or \
                not input_observation_space.is_point_in_space_shape(crop_high - 1):
            raise ValueError("The cropping indices are outside of the observation space")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        # replace -1 with the max size
        crop_high = self._replace_negative_one_in_crop_size(self.crop_high, observation.shape)
        crop_low = self._replace_negative_one_in_crop_size(self.crop_low, observation.shape)

        # crop
        indices = [slice(i, j) for i, j in zip(crop_low, crop_high)]
        observation = observation[indices]
        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        # replace -1 with the max size
        crop_high = self._replace_negative_one_in_crop_size(self.crop_high, input_observation_space.shape)
        crop_low = self._replace_negative_one_in_crop_size(self.crop_low, input_observation_space.shape)

        input_observation_space.shape = crop_high - crop_low
        return input_observation_space
