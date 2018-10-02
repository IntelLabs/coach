import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.graph_managers.graph_manager import GraphManager, ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import RunPhase


@pytest.mark.unit_test
def test_phase_context():
    graph_manager = GraphManager(name='', schedule_params=ScheduleParameters(), vis_params=VisualizationParameters())

    assert graph_manager.phase == RunPhase.UNDEFINED
    with graph_manager.phase_context(RunPhase.TRAIN):
        assert graph_manager.phase == RunPhase.TRAIN
    assert graph_manager.phase == RunPhase.UNDEFINED


@pytest.mark.unit_test
def test_phase_context_nested():
    graph_manager = GraphManager(name='', schedule_params=ScheduleParameters(), vis_params=VisualizationParameters())

    assert graph_manager.phase == RunPhase.UNDEFINED
    with graph_manager.phase_context(RunPhase.TRAIN):
        assert graph_manager.phase == RunPhase.TRAIN
        with graph_manager.phase_context(RunPhase.TEST):
            assert graph_manager.phase == RunPhase.TEST
        assert graph_manager.phase == RunPhase.TRAIN
    assert graph_manager.phase == RunPhase.UNDEFINED


@pytest.mark.unit_test
def test_phase_context_double_nested():
    graph_manager = GraphManager(name='', schedule_params=ScheduleParameters(), vis_params=VisualizationParameters())

    assert graph_manager.phase == RunPhase.UNDEFINED
    with graph_manager.phase_context(RunPhase.TRAIN):
        assert graph_manager.phase == RunPhase.TRAIN
        with graph_manager.phase_context(RunPhase.TEST):
            assert graph_manager.phase == RunPhase.TEST
            with graph_manager.phase_context(RunPhase.UNDEFINED):
                assert graph_manager.phase == RunPhase.UNDEFINED
            assert graph_manager.phase == RunPhase.TEST
        assert graph_manager.phase == RunPhase.TRAIN
    assert graph_manager.phase == RunPhase.UNDEFINED
