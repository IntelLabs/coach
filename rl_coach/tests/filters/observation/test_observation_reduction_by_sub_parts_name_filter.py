import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_reduction_by_sub_parts_name_filter import ObservationReductionBySubPartsNameFilter
from rl_coach.spaces import VectorObservationSpace
from rl_coach.core_types import EnvResponse
from rl_coach.filters.filter import InputFilter


@pytest.mark.unit_test
def test_filter():
    # Keep
    observation_space = VectorObservationSpace(3, measurements_names=['a', 'b', 'c'])
    env_response = EnvResponse(next_state={'observation': np.ones([3])}, reward=0, game_over=False)
    reduction_filter = InputFilter()
    reduction_filter.add_observation_filter('observation', 'reduce',
                                          ObservationReductionBySubPartsNameFilter(
                                              ["a"],
                                              ObservationReductionBySubPartsNameFilter.ReductionMethod.Keep
                                          ))

    reduction_filter.get_filtered_observation_space('observation', observation_space)
    result = reduction_filter.filter(env_response)[0]
    unfiltered_observation = env_response.next_state['observation']
    filtered_observation = result.next_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.shape == (3,)

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (1,)

    # Discard
    reduction_filter = InputFilter()
    reduction_filter.add_observation_filter('observation', 'reduce',
                                          ObservationReductionBySubPartsNameFilter(
                                              ["a"],
                                              ObservationReductionBySubPartsNameFilter.ReductionMethod.Discard
                                          ))
    reduction_filter.get_filtered_observation_space('observation', observation_space)
    result = reduction_filter.filter(env_response)[0]
    unfiltered_observation = env_response.next_state['observation']
    filtered_observation = result.next_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.shape == (3,)

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (2,)


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    # Keep
    observation_space = VectorObservationSpace(3, measurements_names=['a', 'b', 'c'])
    env_response = EnvResponse(next_state={'observation': np.ones([3])}, reward=0, game_over=False)
    reduction_filter = InputFilter()
    reduction_filter.add_observation_filter('observation', 'reduce',
                                            ObservationReductionBySubPartsNameFilter(
                                                ["a"],
                                                ObservationReductionBySubPartsNameFilter.ReductionMethod.Keep
                                            ))

    filtered_observation_space = reduction_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(filtered_observation_space.shape == np.array([1]))
    assert filtered_observation_space.measurements_names == ['a']

    # Discard
    observation_space = VectorObservationSpace(3, measurements_names=['a', 'b', 'c'])
    env_response = EnvResponse(next_state={'observation': np.ones([3])}, reward=0, game_over=False)
    reduction_filter = InputFilter()
    reduction_filter.add_observation_filter('observation', 'reduce',
                                            ObservationReductionBySubPartsNameFilter(
                                                ["a"],
                                                ObservationReductionBySubPartsNameFilter.ReductionMethod.Discard
                                            ))

    filtered_observation_space = reduction_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(filtered_observation_space.shape == np.array([2]))
    assert filtered_observation_space.measurements_names == ['b', 'c']
