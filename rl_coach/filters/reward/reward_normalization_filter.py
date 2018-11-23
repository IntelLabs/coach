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
import os

import numpy as np

from rl_coach.core_types import RewardType
from rl_coach.filters.reward.reward_filter import RewardFilter
from rl_coach.spaces import RewardSpace
from rl_coach.utilities.shared_running_stats import NumpySharedRunningStats


class RewardNormalizationFilter(RewardFilter):
    """
    Normalizes the reward values with a running mean and standard deviation of
    all the rewards seen so far. When working with multiple workers, the statistics used for the normalization operation
    are accumulated over all the workers.
    """
    def __init__(self, clip_min: float=-5.0, clip_max: float=5.0):
        """
        :param clip_min: The minimum value to allow after normalizing the reward
        :param clip_max: The maximum value to allow after normalizing the reward
        """
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.running_rewards_stats = None

    def set_device(self, device, memory_backend_params=None, mode='numpy') -> None:
        """
        An optional function that allows the filter to get the device if it is required to use tensorflow ops
        :param device: the device to use
        :return: None
        """

        if mode == 'tf':
            from rl_coach.architectures.tensorflow_components.shared_variables import TFSharedRunningStats
            self.running_rewards_stats = TFSharedRunningStats(device, name='rewards_stats', create_ops=False,
                                                            pubsub_params=memory_backend_params)
        elif mode == 'numpy':
            self.running_rewards_stats = NumpySharedRunningStats(name='rewards_stats',
                                                          pubsub_params=memory_backend_params)

    def set_session(self, sess) -> None:
        """
        An optional function that allows the filter to get the session if it is required to use tensorflow ops
        :param sess: the session
        :return: None
        """
        self.running_rewards_stats.set_session(sess)

    def filter(self, reward: RewardType, update_internal_state: bool=True) -> RewardType:
        if update_internal_state:
            self.running_rewards_stats.push(reward)

        reward = (reward - self.running_rewards_stats.mean) / \
                      (self.running_rewards_stats.std + 1e-15)
        reward = np.clip(reward, self.clip_min, self.clip_max)

        return reward

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        return input_reward_space

    def save_state_to_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: str):
        self.running_rewards_stats.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)

    def restore_state_from_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: str):
        self.running_rewards_stats.restore_state_from_checkpoint(checkpoint_dir, checkpoint_prefix)
