import mxnet as mx
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.architectures.mxnet_components.heads.ppo_v_head import PPOVHead, PPOVHeadLoss
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAlgorithmParameters, ClippedPPOAgentParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace


@pytest.mark.unit_test
def test_ppo_v_head_loss_batch():
    loss_fn = PPOVHeadLoss(clip_likelihood_ratio_using_epsilon=0.1)
    total_return = mx.nd.array((5, -3, 0))
    old_policy_values = mx.nd.array((3, -1, -1))
    new_policy_values_worse = mx.nd.array((2, 0, -1))
    new_policy_values_better = mx.nd.array((4, -2, -1))

    loss_worse = loss_fn(new_policy_values_worse, old_policy_values, total_return)
    loss_better = loss_fn(new_policy_values_better, old_policy_values, total_return)

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
def test_ppo_v_head_loss_batch_time():
    loss_fn = PPOVHeadLoss(clip_likelihood_ratio_using_epsilon=0.1)
    total_return = mx.nd.array(((3, 1, 1, 0),
                                (1, 0, 0, 1),
                                (3, 0, 1, 0)))
    old_policy_values = mx.nd.array(((2, 1, 1, 0),
                                     (1, 0, 0, 1),
                                     (0, 0, 1, 0)))
    new_policy_values_worse = mx.nd.array(((2, 1, 1, 0),
                                           (1, 0, 0, 1),
                                           (2, 0, 1, 0)))
    new_policy_values_better = mx.nd.array(((3, 1, 1, 0),
                                            (1, 0, 0, 1),
                                            (2, 0, 1, 0)))

    loss_worse = loss_fn(new_policy_values_worse, old_policy_values, total_return)
    loss_better = loss_fn(new_policy_values_better, old_policy_values, total_return)

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
def test_ppo_v_head_loss_weight():
    total_return = mx.nd.array((5, -3, 0))
    old_policy_values = mx.nd.array((3, -1, -1))
    new_policy_values = mx.nd.array((4, -2, -1))
    loss_fn = PPOVHeadLoss(clip_likelihood_ratio_using_epsilon=0.2, weight=1)
    loss = loss_fn(new_policy_values, old_policy_values, total_return)
    loss_fn_weighted = PPOVHeadLoss(clip_likelihood_ratio_using_epsilon=0.2, weight=0.5)
    loss_weighted = loss_fn_weighted(new_policy_values, old_policy_values, total_return)
    assert loss[0].sum() == loss_weighted[0].sum() * 2


@pytest.mark.unit_test
def test_ppo_v_head():
    agent_parameters = ClippedPPOAgentParameters()
    action_space = DiscreteActionSpace(num_actions=5)
    spaces = SpacesDefinition(state=None, goal=None, action=action_space, reward=None)
    value_net = PPOVHead(agent_parameters=agent_parameters,
                         spaces=spaces,
                         network_name="test_ppo_v_head")
    value_net.initialize()
    batch_size = 15
    middleware_data = mx.nd.random.uniform(shape=(batch_size, 100))
    values = value_net(middleware_data)
    assert values.ndim == 1  # (batch_size)
    assert values.shape[0] == batch_size
