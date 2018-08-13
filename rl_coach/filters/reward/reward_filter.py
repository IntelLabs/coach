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
from rl_coach.spaces import RewardSpace


class RewardFilter(Filter):
    def __init__(self):
        super().__init__()

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        """
        This function should contain the logic for getting the filtered reward space
        :param input_reward_space: the input reward space
        :return: the filtered reward space
        """
        return input_reward_space