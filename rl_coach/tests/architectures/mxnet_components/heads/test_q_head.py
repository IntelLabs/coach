import mxnet as mx
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.architectures.mxnet_components.heads.q_head import QHead, QHeadLoss
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace



@pytest.mark.unit_test
def test_q_head_loss():
    loss_fn = QHeadLoss()
    # example with batch_size of 3, and num_actions of 2
    target_q_values = mx.nd.array(((3, 5), (-1, -2), (0, 2)))
    pred_q_values_worse = mx.nd.array(((6, 5), (-1, -2), (0, 2)))
    pred_q_values_better = mx.nd.array(((4, 5), (-2, -2), (1, 2)))
    loss_worse = loss_fn(pred_q_values_worse, target_q_values)
    loss_better = loss_fn(pred_q_values_better, target_q_values)
    assert len(loss_worse) == 1  # (LOSS)
    loss_worse_val = loss_worse[0]
    assert loss_worse_val.ndim == 1
    assert loss_worse_val.shape[0] == 1
    assert len(loss_better) == 1  # (LOSS)
    loss_better_val = loss_better[0]
    assert loss_better_val.ndim == 1
    assert loss_better_val.shape[0] == 1
    assert loss_worse_val > loss_better_val


@pytest.mark.unit_test
def test_v_head_loss_weight():
    target_q_values = mx.nd.array(((3, 5), (-1, -2), (0, 2)))
    pred_q_values = mx.nd.array(((4, 5), (-2, -2), (1, 2)))
    loss_fn = QHeadLoss()
    loss = loss_fn(pred_q_values, target_q_values)
    loss_fn_weighted = QHeadLoss(weight=0.5)
    loss_weighted = loss_fn_weighted(pred_q_values, target_q_values)
    assert loss[0] == loss_weighted[0]*2


@pytest.mark.unit_test
def test_ppo_v_head():
    agent_parameters = ClippedPPOAgentParameters()
    num_actions = 5
    action_space = DiscreteActionSpace(num_actions=num_actions)
    spaces = SpacesDefinition(state=None, goal=None, action=action_space, reward=None)
    value_net = QHead(agent_parameters=agent_parameters,
                      spaces=spaces,
                      network_name="test_q_head")
    value_net.initialize()
    batch_size = 15
    middleware_data = mx.nd.random.uniform(shape=(batch_size, 100))
    values = value_net(middleware_data)
    assert values.ndim == 2  # (batch_size, num_actions)
    assert values.shape[0] == batch_size
    assert values.shape[1] == num_actions