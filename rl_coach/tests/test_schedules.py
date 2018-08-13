import os
import sys

from rl_coach.core_types import EnvironmentSteps

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.schedules import LinearSchedule, ConstantSchedule, ExponentialSchedule, PieceWiseSchedule
import numpy as np


@pytest.mark.unit_test
def test_constant_schedule():
    schedule = ConstantSchedule(0.3)

    # make sure the values in the constant schedule don't change over time
    for i in range(1000):
        assert schedule.initial_value == 0.3
        assert schedule.current_value == 0.3
        schedule.step()


@pytest.mark.unit_test
def test_linear_schedule():
    # increasing schedule
    schedule = LinearSchedule(1, 3, 10)

    # the schedule is defined in number of steps to get from 1 to 3 so there are 10 steps
    # the linspace is defined in number of bins between 1 and 3 so theres are 11 bins
    target_values = np.linspace(1, 3, 11)
    for i in range(10):
        # we round to 4 because there is a very small floating point division difference (1e-10)
        assert round(schedule.current_value, 4) == round(target_values[i], 4)
        schedule.step()

    # make sure the value does not change after 10 steps
    for i in range(10):
        assert schedule.current_value == 3

    # decreasing schedule
    schedule = LinearSchedule(3, 1, 10)

    target_values = np.linspace(3, 1, 11)
    for i in range(10):
        # we round to 4 because there is a very small floating point division difference (1e-10)
        assert round(schedule.current_value, 4) == round(target_values[i], 4)
        schedule.step()

    # make sure the value does not change after 10 steps
    for i in range(10):
        assert schedule.current_value == 1

    # constant schedule
    schedule = LinearSchedule(3, 3, 10)

    for i in range(10):
        # we round to 4 because there is a very small floating point division difference (1e-10)
        assert round(schedule.current_value, 4) == 3
        schedule.step()


@pytest.mark.unit_test
def test_exponential_schedule():
    # decreasing schedule
    schedule = ExponentialSchedule(10, 3, 0.99)

    current_power = 1
    for i in range(100):
        assert round(schedule.current_value,6) == round(10*current_power,6)
        current_power *= 0.99
        schedule.step()

    for i in range(100):
        schedule.step()
    assert schedule.current_value == 3


@pytest.mark.unit_test
def test_piece_wise_schedule():
    # decreasing schedule
    schedule = PieceWiseSchedule(
        [(LinearSchedule(1, 3, 10), EnvironmentSteps(5)),
         (ConstantSchedule(4), EnvironmentSteps(10)),
         (ExponentialSchedule(3, 1, 0.99), EnvironmentSteps(10))
         ]
    )

    target_values = np.append(np.linspace(1, 2, 6), np.ones(11)*4)
    for i in range(16):
        assert round(schedule.current_value, 4) == round(target_values[i], 4)
        schedule.step()

    current_power = 1
    for i in range(10):
        assert round(schedule.current_value, 4) == round(3*current_power, 4)
        current_power *= 0.99
        schedule.step()


if __name__ == "__main__":
    test_constant_schedule()
    test_linear_schedule()
    test_exponential_schedule()
    test_piece_wise_schedule()
