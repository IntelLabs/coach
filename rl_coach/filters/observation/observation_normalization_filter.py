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
from typing import List

import numpy as np

from rl_coach.architectures.tensorflow_components.shared_variables import SharedRunningStats
from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace


class ObservationNormalizationFilter(ObservationFilter):
    """
    Normalize the observation with a running standard deviation and mean of the observations seen so far
    If there is more than a single worker, the statistics of the observations are shared between all the workers
    """
    def __init__(self, clip_min: float=-5.0, clip_max: float=5.0, name='observation_stats'):
        """
        :param clip_min: The minimum value to allow after normalizing the observation
        :param clip_max: The maximum value to allow after normalizing the observation
        """
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.running_observation_stats = None
        self.name = name
        self.supports_batching = True
        self.observation_space = None

    def set_device(self, device) -> None:
        """
        An optional function that allows the filter to get the device if it is required to use tensorflow ops
        :param device: the device to use
        :return: None
        """
        self.running_observation_stats = SharedRunningStats(device, name=self.name, create_ops=False)

    def set_session(self, sess) -> None:
        """
        An optional function that allows the filter to get the session if it is required to use tensorflow ops
        :param sess: the session
        :return: None
        """
        self.running_observation_stats.set_session(sess)

    def filter(self, observations: List[ObservationType], update_internal_state: bool=True) -> ObservationType:
        observations = np.array(observations)
        if update_internal_state:
            self.running_observation_stats.push(observations)
            self.last_mean = self.running_observation_stats.mean
            self.last_stdev = self.running_observation_stats.std

        # TODO: make sure that a batch is given here
        return self.running_observation_stats.normalize(observations)

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        self.running_observation_stats.create_ops(shape=input_observation_space.shape,
                                                  clip_values=(self.clip_min, self.clip_max))
        return input_observation_space
