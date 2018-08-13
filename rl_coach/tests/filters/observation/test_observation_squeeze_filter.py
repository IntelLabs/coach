import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_squeeze_filter import ObservationSqueezeFilter
from rl_coach.spaces import ObservationSpace
from rl_coach.core_types import EnvResponse
from rl_coach.filters.filter import InputFilter


@pytest.mark.unit_test
def test_filter():
    # make an RGB observation smaller
    squeeze_filter = InputFilter()
    squeeze_filter.add_observation_filter('observation', 'squeeze', ObservationSqueezeFilter())
    squeeze_filter_with_axis = InputFilter()
    squeeze_filter_with_axis.add_observation_filter('observation', 'squeeze', ObservationSqueezeFilter(2))

    observation = np.random.rand(20, 30, 1, 3)
    env_response = EnvResponse(next_state={'observation': observation}, reward=0, game_over=False)

    result = squeeze_filter.filter(env_response)[0]
    result_with_axis = squeeze_filter_with_axis.filter(env_response)[0]
    unfiltered_observation_shape = env_response.next_state['observation'].shape
    filtered_observation_shape = result.next_state['observation'].shape
    filtered_observation_with_axis_shape = result_with_axis.next_state['observation'].shape

    # make sure the original observation is unchanged
    assert unfiltered_observation_shape == observation.shape

    # make sure the filtering is done correctly
    assert filtered_observation_shape == (20, 30, 3)
    assert filtered_observation_with_axis_shape == (20, 30, 3)

    observation = np.random.rand(1, 30, 1, 3)
    env_response = EnvResponse(next_state={'observation': observation}, reward=0, game_over=False)

    result = squeeze_filter.filter(env_response)[0]
    assert result.next_state['observation'].shape == (30, 3)


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    # error on observation space with shape not matching the filter squeeze axis configuration
    squeeze_filter = InputFilter()
    squeeze_filter.add_observation_filter('observation', 'squeeze', ObservationSqueezeFilter(axis=3))

    observation_space = ObservationSpace(np.array([20, 1, 30, 3]), 0, 100)
    small_observation_space = ObservationSpace(np.array([20, 1, 30]), 0, 100)
    with pytest.raises(ValueError):
        squeeze_filter.get_filtered_observation_space('observation', observation_space)
        squeeze_filter.get_filtered_observation_space('observation', small_observation_space)

    # verify output observation space is correct
    observation_space = ObservationSpace(np.array([1, 2, 3, 1]), 0, 200)
    result = squeeze_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(result.shape == np.array([1, 2, 3]))

    squeeze_filter = InputFilter()
    squeeze_filter.add_observation_filter('observation', 'squeeze', ObservationSqueezeFilter())

    result = squeeze_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(result.shape == np.array([2, 3]))


if __name__ == '__main__':
    test_filter()
    test_get_filtered_observation_space()

