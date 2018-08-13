import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.exploration_policies.additive_noise import AdditiveNoise
from rl_coach.schedules import LinearSchedule
import numpy as np


@pytest.mark.unit_test
def test_init():
    # discrete control
    action_space = DiscreteActionSpace(3)
    noise_schedule = LinearSchedule(1.0, 1.0, 1000)

    # additive noise doesn't work for discrete controls
    with pytest.raises(ValueError):
        policy = AdditiveNoise(action_space, noise_schedule, 0)

    # additive noise requires a bounded range for the actions
    action_space = BoxActionSpace(np.array([10]))
    with pytest.raises(ValueError):
        policy = AdditiveNoise(action_space, noise_schedule, 0)


@pytest.mark.unit_test
def test_get_action():
    # make sure noise is in range
    action_space = BoxActionSpace(np.array([10]), -1, 1)
    noise_schedule = LinearSchedule(1.0, 1.0, 1000)
    policy = AdditiveNoise(action_space, noise_schedule, 0)

    # the action range is 2, so there is a ~0.1% chance that the noise will be larger than 3*std=3*2=6
    for i in range(1000):
        action = policy.get_action(np.zeros([10]))
        assert np.all(action < 10)
        # make sure there is no clipping of the action since it should be the environment that clips actions
        assert np.all(action != 1.0)
        assert np.all(action != -1.0)
        # make sure that each action element has a different value
        assert np.all(action[0] != action[1:])
