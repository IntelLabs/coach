from rl_coach.core_types import (
    TotalStepsCounter,
    EnvironmentSteps,
    EnvironmentEpisodes,
    StepMethod,
    EnvironmentSteps,
    EnvironmentEpisodes,
)

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


@pytest.mark.unit_test
def test_step_method_div():
    assert StepMethod(10) / 2 == StepMethod(5)
    assert StepMethod(10) / StepMethod(2) == 5


@pytest.mark.unit_test
def test_step_method_div_ceil():
    assert StepMethod(10) / 3 == StepMethod(4)
    assert StepMethod(10) / StepMethod(3) == 4


@pytest.mark.unit_test
def test_step_method_rdiv_ceil():
    assert 10 / StepMethod(3) == StepMethod(4)
    assert StepMethod(10) / StepMethod(3) == 4


@pytest.mark.unit_test
def test_step_method_rdiv():
    assert 10 / StepMethod(2) == StepMethod(5)
    assert StepMethod(10) / StepMethod(2) == 5


@pytest.mark.unit_test
def test_step_method_div_type():
    with pytest.raises(TypeError):
        EnvironmentEpisodes(10) / EnvironmentSteps(2)
