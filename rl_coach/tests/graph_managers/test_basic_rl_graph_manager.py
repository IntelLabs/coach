import gc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tensorflow as tf
from rl_coach.base_parameters import TaskParameters, DistributedTaskParameters, Frameworks
from rl_coach.core_types import EnvironmentSteps
from rl_coach.utils import get_open_port
from multiprocessing import Process
from tensorflow import logging
import pytest
logging.set_verbosity(logging.INFO)


@pytest.mark.unit_test
def test_basic_rl_graph_manager_with_pong_a3c():
    tf.reset_default_graph()
    from rl_coach.presets.Atari_A3C import graph_manager
    assert graph_manager
    graph_manager.env_params.level = "PongDeterministic-v4"
    graph_manager.create_graph(task_parameters=TaskParameters(framework_type=Frameworks.tensorflow,
                                                              experiment_path="./experiments/test"))
    # graph_manager.improve()


@pytest.mark.unit_test
def test_basic_rl_graph_manager_with_pong_nec():
    tf.reset_default_graph()
    from rl_coach.presets.Atari_NEC import graph_manager
    assert graph_manager
    graph_manager.env_params.level = "PongDeterministic-v4"
    graph_manager.create_graph(task_parameters=TaskParameters(framework_type=Frameworks.tensorflow,
                                                              experiment_path="./experiments/test"))
    # graph_manager.improve()


@pytest.mark.unit_test
def test_basic_rl_graph_manager_with_cartpole_dqn():
    tf.reset_default_graph()
    from rl_coach.presets.CartPole_DQN import graph_manager
    assert graph_manager
    graph_manager.create_graph(task_parameters=TaskParameters(framework_type=Frameworks.tensorflow,
                                                              experiment_path="./experiments/test"))
    # graph_manager.improve()

# Test for identifying memory leak in restore_checkpoint
@pytest.mark.unit_test
def test_basic_rl_graph_manager_with_cartpole_dqn_and_repeated_checkpoint_restore():
    tf.reset_default_graph()
    from rl_coach.presets.CartPole_DQN import graph_manager
    assert graph_manager
    graph_manager.create_graph(task_parameters=TaskParameters(framework_type=Frameworks.tensorflow,
                                                              experiment_path="./experiments/test",
                                                              apply_stop_condition=True))
    # graph_manager.improve()
    # graph_manager.evaluate(EnvironmentSteps(1000))
    # graph_manager.save_checkpoint()
    #
    # graph_manager.task_parameters.checkpoint_restore_path = "./experiments/test/checkpoint"
    # while True:
    #     graph_manager.restore_checkpoint()
    #     graph_manager.evaluate(EnvironmentSteps(1000))
    #     gc.collect()

if __name__ == '__main__':
    pass
    # test_basic_rl_graph_manager_with_pong_a3c()
    # test_basic_rl_graph_manager_with_ant_a3c()
    # test_basic_rl_graph_manager_with_pong_nec()
	# test_basic_rl_graph_manager_with_cartpole_dqn()
    # test_basic_rl_graph_manager_with_cartpole_dqn_and_repeated_checkpoint_restore()
    #test_basic_rl_graph_manager_multithreaded_with_pong_a3c()
	#test_basic_rl_graph_manager_with_doom_basic_dqn()