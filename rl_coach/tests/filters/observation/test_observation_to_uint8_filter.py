import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.spaces import ObservationSpace
from rl_coach.core_types import EnvResponse
from rl_coach.filters.filter import InputFilter


@pytest.mark.unit_test
def test_filter():
    # make an RGB observation smaller
    uint8_filter = InputFilter()
    uint8_filter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(input_low=0, input_high=255))

    observation = np.random.rand(20, 30, 3)*255.0
    env_response = EnvResponse(next_state={'observation': observation}, reward=0, game_over=False)

    result = uint8_filter.filter(env_response)[0]
    unfiltered_observation = env_response.next_state['observation']
    filtered_observation = result.next_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.dtype == 'float64'

    # make sure the filtering is done correctly
    assert filtered_observation.dtype == 'uint8'
    assert np.all(filtered_observation == observation.astype('uint8'))


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    # error on observation space with values not matching the filter configuration
    uint8_filter = InputFilter()
    uint8_filter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(input_low=0, input_high=200))

    observation_space = ObservationSpace(np.array([1, 2, 3]), 0, 100)
    with pytest.raises(ValueError):
        uint8_filter.get_filtered_observation_space('observation', observation_space)

    # verify output observation space is correct
    observation_space = ObservationSpace(np.array([1, 2, 3]), 0, 200)
    result = uint8_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(result.high == 255)
    assert np.all(result.low == 0)
    assert np.all(result.shape == observation_space.shape)
