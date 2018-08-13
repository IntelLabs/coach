import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.filters.filter import InputFilter
from rl_coach.spaces import ObservationSpace
from rl_coach.core_types import EnvResponse


@pytest.fixture
def env_response():
    observation = np.random.rand(10, 20, 30)
    return EnvResponse(next_state={'observation': observation}, reward=0, game_over=False)


@pytest.mark.unit_test
def test_filter(env_response):
    crop_low = np.array([0, 5, 10])
    crop_high = np.array([5, 10, 20])
    crop_filter = InputFilter()
    crop_filter.add_observation_filter('observation', 'crop', ObservationCropFilter(crop_low, crop_high))

    result = crop_filter.filter(env_response)[0]
    unfiltered_observation = env_response.next_state['observation']
    filtered_observation = result.next_state['observation']

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (5, 5, 10)

    # validate the content of the filtered observation
    assert np.all(filtered_observation == unfiltered_observation[0:5, 5:10, 10:20])

    # crop with -1 on some axes
    crop_low = np.array([0, 0, 0])
    crop_high = np.array([5, -1, -1])
    crop_filter = InputFilter()
    crop_filter.add_observation_filter('observation', 'crop', ObservationCropFilter(crop_low, crop_high))

    result = crop_filter.filter(env_response)[0]
    unfiltered_observation = env_response.next_state['observation']
    filtered_observation = result.next_state['observation']

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (5, 20, 30)

    # validate the content of the filtered observation
    assert np.all(filtered_observation == unfiltered_observation[0:5, :, :])


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    crop_low = np.array([0, 5, 10])
    crop_high = np.array([5, 10, 20])
    crop_filter = InputFilter()
    crop_filter.add_observation_filter('observation', 'crop', ObservationCropFilter(crop_low, crop_high))

    observation_space = ObservationSpace(np.array([5, 10, 20]))
    filtered_observation_space = crop_filter.get_filtered_observation_space('observation', observation_space)

    # make sure the new observation space shape is calculated correctly
    assert np.all(filtered_observation_space.shape == np.array([5, 5, 10]))

    # make sure the original observation space is unchanged
    assert np.all(observation_space.shape == np.array([5, 10, 20]))

    # crop_high is bigger than the observation space
    high_error_observation_space = ObservationSpace(np.array([3, 8, 14]))
    with pytest.raises(ValueError):
        crop_filter.get_filtered_observation_space('observation', high_error_observation_space)

    # crop_low is bigger than the observation space
    low_error_observation_space = ObservationSpace(np.array([3, 3, 10]))
    with pytest.raises(ValueError):
        crop_filter.get_filtered_observation_space('observation', low_error_observation_space)

    # crop with -1 on some axes
    crop_low = np.array([0, 0, 0])
    crop_high = np.array([5, -1, -1])
    crop_filter = InputFilter()
    crop_filter.add_observation_filter('observation', 'crop', ObservationCropFilter(crop_low, crop_high))

    observation_space = ObservationSpace(np.array([5, 10, 20]))
    filtered_observation_space = crop_filter.get_filtered_observation_space('observation', observation_space)

    # make sure the new observation space shape is calculated correctly
    assert np.all(filtered_observation_space.shape == np.array([5, 10, 20]))
