import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.spaces import RewardSpace
from rl_coach.core_types import EnvResponse
from rl_coach.filters.filter import InputFilter
from collections import OrderedDict


@pytest.mark.unit_test
def test_filter():
    rescale_filter = InputFilter(reward_filters=OrderedDict([('rescale', RewardRescaleFilter(1/10.))]))
    env_response = EnvResponse(next_state={'observation': np.zeros(10)}, reward=100, game_over=False)
    print(rescale_filter.observation_filters)
    result = rescale_filter.filter(env_response)[0]
    unfiltered_reward = env_response.reward
    filtered_reward = result.reward

    # validate that the reward was clipped correctly
    assert filtered_reward == 10

    # make sure the original reward is unchanged
    assert unfiltered_reward == 100

    # negative reward
    env_response = EnvResponse(next_state={'observation': np.zeros(10)}, reward=-50, game_over=False)
    result = rescale_filter.filter(env_response)[0]
    assert result.reward == -5


@pytest.mark.unit_test
def test_get_filtered_reward_space():
    rescale_filter = InputFilter(reward_filters=OrderedDict([('rescale', RewardRescaleFilter(1/10.))]))

    # reward is clipped
    reward_space = RewardSpace(1, -100, 100)
    filtered_reward_space = rescale_filter.get_filtered_reward_space(reward_space)

    # make sure the new reward space shape is calculated correctly
    assert filtered_reward_space.shape == 1
    assert filtered_reward_space.low == -10
    assert filtered_reward_space.high == 10

    # unbounded rewards
    reward_space = RewardSpace(1, -np.inf, np.inf)
    filtered_reward_space = rescale_filter.get_filtered_reward_space(reward_space)

    # make sure the new reward space shape is calculated correctly
    assert filtered_reward_space.shape == 1
    assert filtered_reward_space.low == -np.inf
    assert filtered_reward_space.high == np.inf
