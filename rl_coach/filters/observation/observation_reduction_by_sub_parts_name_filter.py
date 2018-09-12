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
from enum import Enum
from typing import List

import numpy as np

from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace, VectorObservationSpace


class ObservationReductionBySubPartsNameFilter(ObservationFilter):
    """
    Choose sub parts of the observation to remove or keep using their name.
    This is useful when the environment has a measurements vector as observation which includes several different
    measurements, but you want the agent to only see some of the measurements and not all.
    This will currently work only for VectorObservationSpace observations
    """
    class ReductionMethod(Enum):
        Keep = 0
        Discard = 1

    def __init__(self, part_names: List[str], reduction_method: ReductionMethod):
        """
        :param part_names: A list of part names to reduce
        :param reduction_method: A reduction method to use - keep or discard the given parts
        """
        super().__init__()
        self.part_names = part_names
        self.reduction_method = reduction_method
        self.measurement_names = None
        self.indices_to_keep = None

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        if not isinstance(observation, np.ndarray):
            raise ValueError("All the state values are expected to be numpy arrays")
        if self.indices_to_keep is None:
            raise ValueError("To use ObservationReductionBySubPartsNameFilter, the get_filtered_observation_space "
                             "function should be called before filtering an observation")
        observation = observation[..., self.indices_to_keep]
        return observation

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if not isinstance(input_observation_space, VectorObservationSpace):
            raise ValueError("The ObservationReductionBySubPartsNameFilter support only VectorObservationSpace "
                             "observations. The given observation space was: {}"
                             .format(input_observation_space.__class__))

    def get_filtered_observation_space(self, input_observation_space: VectorObservationSpace) -> ObservationSpace:
        self.measurement_names = copy.copy(input_observation_space.measurements_names)

        if self.reduction_method == self.ReductionMethod.Keep:
            input_observation_space.shape[-1] = len(self.part_names)
            self.indices_to_keep = [idx for idx, val in enumerate(self.measurement_names) if val in self.part_names]
            input_observation_space.measurements_names = copy.copy(self.part_names)
        elif self.reduction_method == self.ReductionMethod.Discard:
            input_observation_space.shape[-1] -= len(self.part_names)
            self.indices_to_keep = [idx for idx, val in enumerate(self.measurement_names) if val not in self.part_names]
            input_observation_space.measurements_names = [val for val in input_observation_space.measurements_names if
                                                          val not in self.part_names]
        else:
            raise ValueError("The given reduction method is not supported")

        return input_observation_space
