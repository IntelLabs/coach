import os
import sys

from rl_coach.base_parameters import TaskParameters, Frameworks

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tensorflow as tf
from tensorflow import logging
import pytest
logging.set_verbosity(logging.INFO)


@pytest.mark.unit_test
def test_get_QActionStateValue_predictions():
    tf.reset_default_graph()
    from rl_coach.presets.CartPole_DQN import graph_manager as cartpole_dqn_graph_manager
    assert cartpole_dqn_graph_manager
    cartpole_dqn_graph_manager.create_graph(task_parameters=
                                            TaskParameters(framework_type=Frameworks.tensorflow,
                                                           experiment_path="./experiments/test"))
    cartpole_dqn_graph_manager.improve_steps.num_steps = 1
    cartpole_dqn_graph_manager.steps_between_evaluation_periods.num_steps = 5

    # graph_manager.improve()
    #
    # agent = graph_manager.level_managers[0].composite_agents['simple_rl_agent'].agents['simple_rl_agent/agent']
    # some_state = agent.memory.sample(1)[0].state
    # cartpole_dqn_predictions = agent.get_predictions(states=some_state, prediction_type=QActionStateValue)
    # assert cartpole_dqn_predictions.shape == (1, 2)


if __name__ == '__main__':
    test_get_QActionStateValue_predictions()
