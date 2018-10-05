from rl_coach.core_types import TotalStepsCounter, EnvironmentSteps, EnvironmentEpisodes

import pytest


@pytest.mark.unit_test
def test_add_total_steps_counter():
    counter = TotalStepsCounter()
    steps = counter + EnvironmentSteps(10)
    assert steps.num_steps == 10


@pytest.mark.unit_test
def test_add_total_steps_counter_non_zero():
    counter = TotalStepsCounter()
    counter[EnvironmentSteps] += 10
    steps = counter + EnvironmentSteps(10)
    assert steps.num_steps == 20


@pytest.mark.unit_test
def test_total_steps_counter_less_than():
    counter = TotalStepsCounter()
    steps = counter + EnvironmentSteps(0)
    assert not (counter < steps)
    steps = counter + EnvironmentSteps(1)
    assert counter < steps
