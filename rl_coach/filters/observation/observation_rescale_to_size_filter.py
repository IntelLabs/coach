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
from skimage.transform import resize
import numpy as np

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace, PlanarMapsObservationSpace, ImageObservationSpace


class ObservationRescaleToSizeFilter(ObservationFilter):
    """
    Rescales an image observation to a given size. The target size does not
    necessarily keep the aspect ratio of the original observation.
    Warning: this requires the input observation to be of type uint8 due to scipy requirements!
    """
    def __init__(self, output_observation_space: PlanarMapsObservationSpace):
        """
        :param output_observation_space: the output observation space
        """
        super().__init__()
        self.output_observation_space = output_observation_space

        if not isinstance(output_observation_space, PlanarMapsObservationSpace):
            raise ValueError("The rescale filter only applies to observation spaces that inherit from "
                             "PlanarMapsObservationSpace. This includes observations which consist of a set of 2D "
                             "images or an RGB image. Instead the output observation space was defined as: {}"
                             .format(output_observation_space.__class__))

        self.planar_map_output_shape = copy.copy(self.output_observation_space.shape)
        self.planar_map_output_shape = np.delete(self.planar_map_output_shape,
                                                 self.output_observation_space.channels_axis)

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if not isinstance(input_observation_space, PlanarMapsObservationSpace):
            raise ValueError("The rescale filter only applies to observation spaces that inherit from "
                             "PlanarMapsObservationSpace. This includes observations which consist of a set of 2D "
                             "images or an RGB image. Instead the input observation space was defined as: {}"
                             .format(input_observation_space.__class__))
        if input_observation_space.shape[input_observation_space.channels_axis] \
                != self.output_observation_space.shape[self.output_observation_space.channels_axis]:
            raise ValueError("The number of channels between the input and output observation spaces must match. "
                             "Instead the number of channels were: {}, {}"
                             .format(input_observation_space.shape[input_observation_space.channels_axis],
                             self.output_observation_space.shape[self.output_observation_space.channels_axis]))

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        observation = observation.astype('uint8')

        # rescale
        if isinstance(self.output_observation_space, ImageObservationSpace):
            observation = resize(observation, tuple(self.output_observation_space.shape), anti_aliasing=False,
                                 preserve_range=True).astype('uint8')

        else:
            new_observation = []
            for i in range(self.output_observation_space.shape[self.output_observation_space.channels_axis]):
                new_observation.append(resize(observation.take(i, self.output_observation_space.channels_axis),
                                              tuple(self.planar_map_output_shape),
                                              preserve_range=True).astype('uint8'))
            new_observation = np.array(new_observation)
            observation = new_observation.swapaxes(0, self.output_observation_space.channels_axis)

        return observation

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape = self.output_observation_space.shape
        return input_observation_space
