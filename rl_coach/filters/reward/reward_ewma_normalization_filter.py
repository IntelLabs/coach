#
# Copyright (c) 2019 Intel Corporation
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
import os

import numpy as np
import pickle

from rl_coach.core_types import RewardType
from rl_coach.filters.reward.reward_filter import RewardFilter
from rl_coach.spaces import RewardSpace
from rl_coach.utils import get_latest_checkpoint


class RewardEwmaNormalizationFilter(RewardFilter):
    """
    Normalizes the reward values based on Exponential Weighted Moving Average.   
    """
    def __init__(self, alpha: float = 0.01):
        """
        :param alpha: the degree of weighting decrease, a constant smoothing factor between 0 and 1.
                      A higher alpha discounts older observations faster
        """
        super().__init__()
        self.alpha = alpha
        self.moving_average = 0
        self.initialized = False
        self.checkpoint_file_extension = 'ewma'
        self.supports_batching = True

    def filter(self, reward: RewardType, update_internal_state: bool=True) -> RewardType:
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)

        if update_internal_state:
            mean_rewards = np.mean(reward)

            if not self.initialized:
                self.moving_average = mean_rewards
                self.initialized = True
            else:
                self.moving_average += self.alpha * (mean_rewards - self.moving_average)

        return reward - self.moving_average

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        return input_reward_space

    def save_state_to_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: int):
        dict_to_save = {'moving_average': self.moving_average}

        with open(os.path.join(checkpoint_dir, str(checkpoint_prefix) + '.' + self.checkpoint_file_extension), 'wb') as f:
            pickle.dump(dict_to_save, f, pickle.HIGHEST_PROTOCOL)

    def restore_state_from_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: str):
        latest_checkpoint_filename = get_latest_checkpoint(checkpoint_dir, checkpoint_prefix,
                                                           self.checkpoint_file_extension)

        if latest_checkpoint_filename == '':
            raise ValueError("Could not find RewardEwmaNormalizationFilter checkpoint file. ")

        with open(os.path.join(checkpoint_dir, str(latest_checkpoint_filename)), 'rb') as f:
            saved_dict = pickle.load(f)
            self.__dict__.update(saved_dict)
