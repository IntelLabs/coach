import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.spaces import ObservationSpace
from rl_coach.core_types import EnvResponse

from rl_coach.filters.filter import InputFilter

@pytest.fixture
def rgb_to_y_filter():
    rgb_to_y_filter = InputFilter()
    rgb_to_y_filter.add_observation_filter('observation', 'rgb_to_y', ObservationRGBToYFilter())
    return rgb_to_y_filter


@pytest.mark.unit_test
def test_filter(rgb_to_y_filter):
    # convert RGB observation to graysacle
    observation = np.random.rand(20, 30, 3)*255.0
    transition = EnvResponse(next_state={'observation': observation}, reward=0, game_over=False)

    result = rgb_to_y_filter.filter(transition)[0]
    unfiltered_observation = transition.next_state['observation']
    filtered_observation = result.next_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.shape == (20, 30, 3)

    # make sure the filtering is done correctly
    assert filtered_observation.shape == (20, 30)


@pytest.mark.unit_test
def test_get_filtered_observation_space(rgb_to_y_filter):
    # error on observation space which are not RGB
    observation_space = ObservationSpace(np.array([1, 2, 4]), 0, 100)
    with pytest.raises(ValueError):
        rgb_to_y_filter.get_filtered_observation_space('observation', observation_space)

    observation_space = ObservationSpace(np.array([1, 2, 3]), 0, 100)
    result = rgb_to_y_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(result.shape == np.array([1, 2]))
