import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter, RescaleInterpolationType
from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.filter import InputFilter
from rl_coach.spaces import ImageObservationSpace
import numpy as np
from rl_coach.core_types import EnvResponse
from collections import OrderedDict


@pytest.mark.filterwarnings('ignore:Conversion of')
@pytest.mark.unit_test
def test_filter_stacking():
    # test that filter stacking works fine by taking as input a transition with:
    # - an observation of shape 210x160,
    # - a reward of 100
    # filtering it by:
    # - rescaling the observation to 110x84
    # - cropping the observation to 84x84
    # - clipping the reward to 1
    # - stacking 4 observations to get 84x84x4

    env_response = EnvResponse({'observation': np.ones([210, 160])}, reward=100, game_over=False)

    filter1 = ObservationRescaleToSizeFilter(
        output_observation_space=ImageObservationSpace(np.array([110, 84]), high=255),
        rescaling_interpolation_type=RescaleInterpolationType.BILINEAR
    )

    filter2 = ObservationCropFilter(
        crop_low=np.array([16, 0]),
        crop_high=np.array([100, 84])
    )

    filter3 = RewardClippingFilter(
        clipping_low=-1,
        clipping_high=1
    )

    output_filter = ObservationStackingFilter(
        stack_size=4,
        stacking_axis=-1
    )

    input_filter = InputFilter(
        observation_filters={
            "observation": OrderedDict([
                ("filter1", filter1),
                ("filter2", filter2),
                ("output_filter", output_filter)
            ])},
        reward_filters=OrderedDict([
            ("filter3", filter3)
        ])
    )

    result = input_filter.filter(env_response)[0]
    observation = np.array(result.next_state['observation'])
    assert observation.shape == (84, 84, 4)
    assert np.all(observation == np.ones([84, 84, 4]))
    assert result.reward == 1


