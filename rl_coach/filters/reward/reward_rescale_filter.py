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

from rl_coach.core_types import RewardType
from rl_coach.filters.reward.reward_filter import RewardFilter
from rl_coach.spaces import RewardSpace


class RewardRescaleFilter(RewardFilter):
    """
    Rescales the reward by a given factor. Rescaling the rewards of the environment has been
    observed to have a large effect (negative or positive) on the behavior of the learning process.
    """
    def __init__(self, rescale_factor: float):
        """
        :param rescale_factor: The reward rescaling factor by which the reward will be multiplied
        """
        super().__init__()
        self.rescale_factor = rescale_factor

        if rescale_factor == 0:
            raise ValueError("The reward rescale value can not be set to 0")

    def filter(self, reward: RewardType, update_internal_state: bool=True) -> RewardType:
        reward = float(reward) * self.rescale_factor
        return reward

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        input_reward_space.high = input_reward_space.high * self.rescale_factor
        input_reward_space.low = input_reward_space.low * self.rescale_factor
        return input_reward_space
