import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter, RescaleInterpolationType
from rl_coach.spaces import ObservationSpace, ImageObservationSpace, PlanarMapsObservationSpace
from rl_coach.core_types import EnvResponse
from rl_coach.filters.filter import InputFilter


@pytest.mark.filterwarnings('ignore:Conversion of')
@pytest.mark.unit_test
def test_filter():
    # make an RGB observation smaller
    transition = EnvResponse(next_state={'observation': np.ones([20, 30, 3])}, reward=0, game_over=False)
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([10, 20, 3]),
                                                                                              high=255),
                                                    RescaleInterpolationType.BILINEAR))

    result = rescale_filter.filter(transition)[0]
    unfiltered_observation = transition.next_state['observation']
    filtered_observation = result.next_state['observation']

    # make sure the original observation is unchanged
    assert unfiltered_observation.shape == (20, 30, 3)

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (10, 20, 3)
    assert np.all(filtered_observation == np.ones([10, 20, 3]))

    # make a grayscale observation bigger
    transition = EnvResponse(next_state={'observation': np.ones([20, 30])}, reward=0, game_over=False)
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([40, 60]),
                                                                                              high=255),
                                                    RescaleInterpolationType.BILINEAR))
    result = rescale_filter.filter(transition)[0]
    filtered_observation = result.next_state['observation']

    # validate the shape of the filtered observation
    assert filtered_observation.shape == (40, 60)
    assert np.all(filtered_observation == np.ones([40, 60]))

    # rescale channels -> error
    # with pytest.raises(ValueError):
    #     InputFilter(
    #         observation_filters=OrderedDict([('rescale',
    #                                          ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([10, 20, 1]),
    #                                                                                               high=255),
    #                                                                         RescaleInterpolationType.BILINEAR))]))

    # TODO: validate input to filter
    # different number of axes -> error
    # env_response = EnvResponse(state={'observation': np.ones([20, 30, 3])}, reward=0, game_over=False)
    # rescale_filter = ObservationRescaleToSizeFilter(ObservationSpace(np.array([10, 20])),
    #                                                 RescaleInterpolationType.BILINEAR)
    # with pytest.raises(ValueError):
    #     result = rescale_filter.filter(transition)

    # channels first -> error
    with pytest.raises(ValueError):
        ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([3, 10, 20]), high=255),
                                       RescaleInterpolationType.BILINEAR)


@pytest.mark.unit_test
def test_get_filtered_observation_space():
    # error on wrong number of channels
    with pytest.raises(ValueError):
        observation_filters = InputFilter()
        observation_filters.add_observation_filter('observation', 'rescale',
                                             ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([5, 10, 5]),
                                                                                                  high=255),
                                                                            RescaleInterpolationType.BILINEAR))

    # mismatch and wrong number of channels
    rescale_filter = InputFilter()
    rescale_filter.add_observation_filter('observation', 'rescale',
                                         ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([5, 10, 3]),
                                                                                              high=255),
                                                    RescaleInterpolationType.BILINEAR))

    observation_space = PlanarMapsObservationSpace(np.array([10, 20, 5]), low=0, high=255)
    with pytest.raises(ValueError):
        rescale_filter.get_filtered_observation_space('observation', observation_space)

    # error on wrong number of dimensions
    observation_space = ObservationSpace(np.array([10, 20, 10, 3]), high=255)
    with pytest.raises(ValueError):
        rescale_filter.get_filtered_observation_space('observation', observation_space)

    # make sure the new observation space shape is calculated correctly
    observation_space = ImageObservationSpace(np.array([10, 20, 3]), high=255)
    filtered_observation_space = rescale_filter.get_filtered_observation_space('observation', observation_space)
    assert np.all(filtered_observation_space.shape == np.array([5, 10, 3]))

    # make sure the original observation space is unchanged
    assert np.all(observation_space.shape == np.array([10, 20, 3]))

    # TODO: test that the type of the observation space stays the same
