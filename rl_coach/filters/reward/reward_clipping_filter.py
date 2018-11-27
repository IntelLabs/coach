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

import numpy as np

from rl_coach.core_types import RewardType
from rl_coach.filters.reward.reward_filter import RewardFilter
from rl_coach.spaces import RewardSpace


class RewardClippingFilter(RewardFilter):
    """
    Clips the reward values into a given range. For example, in DQN, the Atari rewards are
    clipped into the range -1 and 1 in order to control the scale of the returns.
    """
    def __init__(self, clipping_low: float=-np.inf, clipping_high: float=np.inf):
        """
        :param clipping_low: The low threshold for reward clipping
        :param clipping_high: The high threshold for reward clipping
        """
        super().__init__()
        self.clipping_low = clipping_low
        self.clipping_high = clipping_high

        if clipping_low > clipping_high:
            raise ValueError("The reward clipping low must be lower than the reward clipping max")

    def filter(self, reward: RewardType, update_internal_state: bool=True) -> RewardType:
        reward = float(reward)

        if self.clipping_high:
            reward = min(reward, self.clipping_high)
        if self.clipping_low:
            reward = max(reward, self.clipping_low)

        return reward

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        input_reward_space.high = min(self.clipping_high, input_reward_space.high)
        input_reward_space.low = max(self.clipping_low, input_reward_space.low)
        return input_reward_space
