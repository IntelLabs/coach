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

from rl_coach.filters.filter import Filter
from rl_coach.spaces import ObservationSpace


class ObservationFilter(Filter):
    def __init__(self):
        super().__init__()
        self.supports_batching = False

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        """
        This function should contain the logic for getting the filtered observation space
        :param input_observation_space: the input observation space
        :return: the filtered observation space
        """
        return input_observation_space

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        """
        A function that implements validation of the input observation space
        :param input_observation_space: the input observation space
        :return: None
        """
        pass