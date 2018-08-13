import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.spaces import RewardSpace
from rl_coach.core_types import EnvResponse

from collections import OrderedDict
from rl_coach.filters.filter import InputFilter


@pytest.fixture
def clip_filter():
    return InputFilter(reward_filters=OrderedDict([('clip', RewardClippingFilter(2, 10))]))


@pytest.mark.unit_test
def test_filter(clip_filter):
    transition = EnvResponse(next_state={'observation': np.zeros(10)}, reward=100, game_over=False)
    result = clip_filter.filter(transition)[0]
    unfiltered_reward = transition.reward
    filtered_reward = result.reward

    # validate that the reward was clipped correctly
    assert filtered_reward == 10

    # make sure the original reward is unchanged
    assert unfiltered_reward == 100

    # reward in bounds
    transition = EnvResponse(next_state={'observation': np.zeros(10)}, reward=5, game_over=False)
    result = clip_filter.filter(transition)[0]
    assert result.reward == 5

    # reward below bounds
    transition = EnvResponse(next_state={'observation': np.zeros(10)}, reward=-5, game_over=False)
    result = clip_filter.filter(transition)[0]
    assert result.reward == 2


@pytest.mark.unit_test
def test_get_filtered_reward_space(clip_filter):
    # reward is clipped
    reward_space = RewardSpace(1, -100, 100)
    filtered_reward_space = clip_filter.get_filtered_reward_space(reward_space)

    # make sure the new reward space shape is calculated correctly
    assert filtered_reward_space.shape == 1
    assert filtered_reward_space.low == 2
    assert filtered_reward_space.high == 10

    # reward is unclipped
    reward_space = RewardSpace(1, 5, 7)
    filtered_reward_space = clip_filter.get_filtered_reward_space(reward_space)

    # make sure the new reward space shape is calculated correctly
    assert filtered_reward_space.shape == 1
    assert filtered_reward_space.low == 5
    assert filtered_reward_space.high == 7

    # infinite reward is clipped
    reward_space = RewardSpace(1, -np.inf, np.inf)
    filtered_reward_space = clip_filter.get_filtered_reward_space(reward_space)

    # make sure the new reward space shape is calculated correctly
    assert filtered_reward_space.shape == 1
    assert filtered_reward_space.low == 2
    assert filtered_reward_space.high == 10


