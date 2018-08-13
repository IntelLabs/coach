import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.spaces import DiscreteActionSpace
from rl_coach.exploration_policies.e_greedy import EGreedy
from rl_coach.schedules import LinearSchedule
import numpy as np
from rl_coach.core_types import RunPhase


@pytest.mark.unit_test
def test_get_action():
    # discrete control
    action_space = DiscreteActionSpace(3)
    epsilon_schedule = LinearSchedule(1.0, 1.0, 1000)
    policy = EGreedy(action_space, epsilon_schedule, evaluation_epsilon=0)

    # verify that test phase gives greedy actions (evaluation_epsilon = 0)
    policy.change_phase(RunPhase.TEST)
    for i in range(100):
        best_action = policy.get_action(np.array([10, 20, 30]))
        assert best_action == 2

    # verify that train phase gives uniform actions (exploration = 1)
    policy.change_phase(RunPhase.TRAIN)
    counters = np.array([0, 0, 0])
    for i in range(30000):
        best_action = policy.get_action(np.array([10, 20, 30]))
        counters[best_action] += 1
    assert np.all(counters > 9500)  # this is noisy so we allow 5% error

    # TODO: test continuous actions


@pytest.mark.unit_test
def test_change_phase():
    # discrete control
    action_space = DiscreteActionSpace(3)
    epsilon_schedule = LinearSchedule(1.0, 0.1, 1000)
    policy = EGreedy(action_space, epsilon_schedule, evaluation_epsilon=0.01)

    # verify schedule not applying if not in training phase
    assert policy.get_control_param() == 1.0
    policy.change_phase(RunPhase.TEST)
    best_action = policy.get_action(np.array([10, 20, 30]))
    assert policy.epsilon_schedule.current_value == 1.0
    policy.change_phase(RunPhase.HEATUP)
    best_action = policy.get_action(np.array([10, 20, 30]))
    assert policy.epsilon_schedule.current_value == 1.0
    policy.change_phase(RunPhase.UNDEFINED)
    best_action = policy.get_action(np.array([10, 20, 30]))
    assert policy.epsilon_schedule.current_value == 1.0


@pytest.mark.unit_test
def test_get_control_param():
    # discrete control
    action_space = DiscreteActionSpace(3)
    epsilon_schedule = LinearSchedule(1.0, 0.1, 1000)
    policy = EGreedy(action_space, epsilon_schedule, evaluation_epsilon=0.01)

    # verify schedule applies to TRAIN phase
    policy.change_phase(RunPhase.TRAIN)
    for i in range(999):
        best_action = policy.get_action(np.array([10, 20, 30]))
        assert 1.0 > policy.get_control_param() > 0.1
    best_action = policy.get_action(np.array([10, 20, 30]))
    assert policy.get_control_param() == 0.1

    # test phases
    policy.change_phase(RunPhase.TEST)
    assert policy.get_control_param() == 0.01

    policy.change_phase(RunPhase.TRAIN)
    assert policy.get_control_param() == 0.1

    policy.change_phase(RunPhase.HEATUP)
    assert policy.get_control_param() == 0.1
