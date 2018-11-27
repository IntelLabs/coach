import mxnet as mx
import numpy as np
import os
import pytest
from scipy import stats as sp_stats
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.architectures.head_parameters import PPOHeadParameters
from rl_coach.architectures.mxnet_components.heads.ppo_head import CategoricalDist, MultivariateNormalDist,\
    DiscretePPOHead, ClippedPPOLossDiscrete, ClippedPPOLossContinuous, PPOHead
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAlgorithmParameters, ClippedPPOAgentParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace


@pytest.mark.unit_test
def test_multivariate_normal_dist_shape():
    num_var = 2
    means = mx.nd.array((0, 1))
    covar = mx.nd.array(((1, 0),(0, 0.5)))
    data = mx.nd.array((0.5, 0.8))
    policy_dist = MultivariateNormalDist(num_var, means, covar)
    log_probs = policy_dist.log_prob(data)
    assert log_probs.ndim == 1
    assert log_probs.shape[0] == 1


@pytest.mark.unit_test
def test_multivariate_normal_dist_batch_shape():
    num_var = 2
    batch_size = 3
    means = mx.nd.random.uniform(shape=(batch_size, num_var))
    # create batch of covariance matrices only defined on diagonal
    std = mx.nd.array((1, 0.5)).broadcast_like(means).expand_dims(-2)
    covar = mx.nd.eye(N=num_var) * std
    data = mx.nd.random.uniform(shape=(batch_size, num_var))
    policy_dist = MultivariateNormalDist(num_var, means, covar)
    log_probs = policy_dist.log_prob(data)
    assert log_probs.ndim == 1
    assert log_probs.shape[0] == batch_size


@pytest.mark.unit_test
def test_multivariate_normal_dist_batch_time_shape():
    num_var = 2
    batch_size = 3
    time_steps = 4
    means = mx.nd.random.uniform(shape=(batch_size, time_steps, num_var))
    # create batch (per time step) of covariance matrices only defined on diagonal
    std = mx.nd.array((1, 0.5)).broadcast_like(means).expand_dims(-2)
    covar = mx.nd.eye(N=num_var) * std
    data = mx.nd.random.uniform(shape=(batch_size, time_steps, num_var))
    policy_dist = MultivariateNormalDist(num_var, means, covar)
    log_probs = policy_dist.log_prob(data)
    assert log_probs.ndim == 2
    assert log_probs.shape[0] == batch_size
    assert log_probs.shape[1] == time_steps


@pytest.mark.unit_test
def test_multivariate_normal_dist_kl_div():
    n_classes = 2
    dist_a = MultivariateNormalDist(num_var=n_classes,
                                    mean = mx.nd.array([0.2, 0.8]).expand_dims(0),
                                    sigma = mx.nd.array([[1, 0.5], [0.5, 0.5]]).expand_dims(0))
    dist_b = MultivariateNormalDist(num_var=n_classes,
                                    mean = mx.nd.array([0.3, 0.7]).expand_dims(0),
                                    sigma = mx.nd.array([[1, 0.2], [0.2, 0.5]]).expand_dims(0))

    actual = dist_a.kl_div(dist_b).asnumpy()
    np.testing.assert_almost_equal(actual=actual, desired=0.195100128)


@pytest.mark.unit_test
def test_multivariate_normal_dist_kl_div_batch():
    n_classes = 2
    dist_a = MultivariateNormalDist(num_var=n_classes,
                                    mean = mx.nd.array([[0.2, 0.8],
                                                        [0.2, 0.8]]),
                                    sigma = mx.nd.array([[[1, 0.5], [0.5, 0.5]],
                                                         [[1, 0.5], [0.5, 0.5]]]))
    dist_b = MultivariateNormalDist(num_var=n_classes,
                                    mean = mx.nd.array([[0.3, 0.7],
                                                        [0.3, 0.7]]),
                                    sigma = mx.nd.array([[[1, 0.2], [0.2, 0.5]],
                                                         [[1, 0.2], [0.2, 0.5]]]))

    actual = dist_a.kl_div(dist_b).asnumpy()
    np.testing.assert_almost_equal(actual=actual, desired=[0.195100128, 0.195100128])


@pytest.mark.unit_test
def test_categorical_dist_shape():
    num_actions = 2
    # actions taken, of shape (batch_size, time_steps)
    actions = mx.nd.array((1,))
    # action probabilities, of shape (batch_size, time_steps, num_actions)
    policy_probs = mx.nd.array((0.8, 0.2))
    policy_dist = CategoricalDist(num_actions, policy_probs)
    action_probs = policy_dist.log_prob(actions)
    assert action_probs.ndim == 1
    assert action_probs.shape[0] == 1


@pytest.mark.unit_test
def test_categorical_dist_batch_shape():
    batch_size = 3
    num_actions = 2
    # actions taken, of shape (batch_size, time_steps)
    actions = mx.nd.array((0, 1, 0))
    # action probabilities, of shape (batch_size, time_steps, num_actions)
    policy_probs = mx.nd.array(((0.8, 0.2), (0.5, 0.5), (0.5, 0.5)))
    policy_dist = CategoricalDist(num_actions, policy_probs)
    action_probs = policy_dist.log_prob(actions)
    assert action_probs.ndim == 1
    assert action_probs.shape[0] == batch_size


@pytest.mark.unit_test
def test_categorical_dist_batch_time_shape():
    batch_size = 3
    time_steps = 4
    num_actions = 2
    # actions taken, of shape (batch_size, time_steps)
    actions = mx.nd.array(((0, 1, 0, 0),
                           (1, 1, 0, 0),
                           (0, 0, 0, 0)))
    # action probabilities, of shape (batch_size, time_steps, num_actions)
    policy_probs = mx.nd.array((((0.8, 0.2), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))))
    policy_dist = CategoricalDist(num_actions, policy_probs)
    action_probs = policy_dist.log_prob(actions)
    assert action_probs.ndim == 2
    assert action_probs.shape[0] == batch_size
    assert action_probs.shape[1] == time_steps


@pytest.mark.unit_test
def test_categorical_dist_batch():
    n_classes = 2
    probs = mx.nd.array(((0.8, 0.2),
                         (0.7, 0.3),
                         (0.5, 0.5)))

    dist = CategoricalDist(n_classes, probs)
    # check log_prob
    actions = mx.nd.array((0, 1, 0))
    manual_log_prob = np.array((-0.22314353, -1.20397282, -0.69314718))
    np.testing.assert_almost_equal(actual=dist.log_prob(actions).asnumpy(), desired=manual_log_prob)
    # check entropy
    sp_entropy = np.array([sp_stats.entropy(pk=(0.8, 0.2)),
                           sp_stats.entropy(pk=(0.7, 0.3)),
                           sp_stats.entropy(pk=(0.5, 0.5))])
    np.testing.assert_almost_equal(actual=dist.entropy().asnumpy(), desired=sp_entropy)


@pytest.mark.unit_test
def test_categorical_dist_kl_div():
    n_classes = 3
    dist_a = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([0.4, 0.2, 0.4]))
    dist_b = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([0.3, 0.4, 0.3]))
    dist_c = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([0.2, 0.6, 0.2]))
    dist_d = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([0.0, 1.0, 0.0]))
    np.testing.assert_almost_equal(actual=dist_a.kl_div(dist_b).asnumpy(), desired=0.09151624)
    np.testing.assert_almost_equal(actual=dist_a.kl_div(dist_c).asnumpy(), desired=0.33479536)
    np.testing.assert_almost_equal(actual=dist_c.kl_div(dist_a).asnumpy(), desired=0.38190854)
    np.testing.assert_almost_equal(actual=dist_a.kl_div(dist_d).asnumpy(), desired=np.nan)
    np.testing.assert_almost_equal(actual=dist_d.kl_div(dist_a).asnumpy(), desired=1.60943782)


@pytest.mark.unit_test
def test_categorical_dist_kl_div_batch():
    n_classes = 3
    dist_a = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([[0.4, 0.2, 0.4],
                                                                     [0.4, 0.2, 0.4],
                                                                     [0.4, 0.2, 0.4]]))
    dist_b = CategoricalDist(n_classes=n_classes, probs=mx.nd.array([[0.3, 0.4, 0.3],
                                                                     [0.3, 0.4, 0.3],
                                                                     [0.3, 0.4, 0.3]]))
    actual = dist_a.kl_div(dist_b).asnumpy()
    np.testing.assert_almost_equal(actual=actual, desired=[0.09151624, 0.09151624, 0.09151624])


@pytest.mark.unit_test
def test_clipped_ppo_loss_continuous_batch():
    # check lower loss for policy with better probabilities:
    # i.e. higher probability on high advantage actions, low probability on low advantage actions.
    loss_fn = ClippedPPOLossContinuous(num_actions=2,
                                       clip_likelihood_ratio_using_epsilon=0.2)
    loss_fn.initialize()
    # actual actions taken, of shape (batch_size)
    actions = mx.nd.array(((0.5, -0.5), (0.2, 0.3), (0.4, 2.0)))
    # advantages from taking action, of shape (batch_size)
    advantages = mx.nd.array((2, -2, 1))
    # action probabilities, of shape (batch_size, num_actions)
    old_policy_means = mx.nd.array(((1, 0), (0, 0), (-1, 0)))
    new_policy_means_worse = mx.nd.array(((2, 0), (0, 0), (-1, 0)))
    new_policy_means_better = mx.nd.array(((0.5, 0), (0, 0), (-1, 0)))

    policy_stds = mx.nd.array(((1, 1), (1, 1), (1, 1)))
    clip_param_rescaler = mx.nd.array((1,))

    loss_worse = loss_fn(new_policy_means_worse, policy_stds,
                         actions, old_policy_means, policy_stds,
                         clip_param_rescaler, advantages)
    loss_better = loss_fn(new_policy_means_better, policy_stds,
                          actions, old_policy_means, policy_stds,
                          clip_param_rescaler, advantages)

    assert len(loss_worse) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    loss_worse_val = loss_worse[0]
    assert loss_worse_val.ndim == 1
    assert loss_worse_val.shape[0] == 1
    assert len(loss_better) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    loss_better_val = loss_better[0]
    assert loss_better_val.ndim == 1
    assert loss_better_val.shape[0] == 1
    assert loss_worse_val > loss_better_val


@pytest.mark.unit_test
def test_clipped_ppo_loss_discrete_batch():
    # check lower loss for policy with better probabilities:
    # i.e. higher probability on high advantage actions, low probability on low advantage actions.
    loss_fn = ClippedPPOLossDiscrete(num_actions=2,
                                     clip_likelihood_ratio_using_epsilon=None,
                                     use_kl_regularization=True,
                                     initial_kl_coefficient=1)
    loss_fn.initialize()

    # actual actions taken, of shape (batch_size)
    actions = mx.nd.array((0, 1, 0))
    # advantages from taking action, of shape (batch_size)
    advantages = mx.nd.array((-2, 2, 1))
    # action probabilities, of shape (batch_size, num_actions)
    old_policy_probs = mx.nd.array(((0.7, 0.3), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs_worse = mx.nd.array(((0.9, 0.1), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs_better = mx.nd.array(((0.5, 0.5), (0.2, 0.8), (0.4, 0.6)))

    clip_param_rescaler = mx.nd.array((1,))

    loss_worse = loss_fn(new_policy_probs_worse, actions, old_policy_probs, clip_param_rescaler, advantages)
    loss_better = loss_fn(new_policy_probs_better, actions, old_policy_probs, clip_param_rescaler, advantages)

    assert len(loss_worse) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    lw_loss, lw_reg, lw_kl, lw_ent, lw_lr, lw_clip_lr = loss_worse
    assert lw_loss.ndim == 1
    assert lw_loss.shape[0] == 1
    assert len(loss_better) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    lb_loss, lb_reg, lb_kl, lb_ent, lb_lr, lb_clip_lr = loss_better
    assert lb_loss.ndim == 1
    assert lb_loss.shape[0] == 1
    assert lw_loss > lb_loss
    assert lw_kl > lb_kl


@pytest.mark.unit_test
def test_clipped_ppo_loss_discrete_batch_kl_div():
    # check lower loss for policy with better probabilities:
    # i.e. higher probability on high advantage actions, low probability on low advantage actions.
    loss_fn = ClippedPPOLossDiscrete(num_actions=2,
                                     clip_likelihood_ratio_using_epsilon=None,
                                     use_kl_regularization=True,
                                     initial_kl_coefficient=0.5)
    loss_fn.initialize()

    # actual actions taken, of shape (batch_size)
    actions = mx.nd.array((0, 1, 0))
    # advantages from taking action, of shape (batch_size)
    advantages = mx.nd.array((-2, 2, 1))
    # action probabilities, of shape (batch_size, num_actions)
    old_policy_probs = mx.nd.array(((0.7, 0.3), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs_worse = mx.nd.array(((0.9, 0.1), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs_better = mx.nd.array(((0.5, 0.5), (0.2, 0.8), (0.4, 0.6)))

    clip_param_rescaler = mx.nd.array((1,))

    loss_worse = loss_fn(new_policy_probs_worse, actions, old_policy_probs, clip_param_rescaler, advantages)
    loss_better = loss_fn(new_policy_probs_better, actions, old_policy_probs, clip_param_rescaler, advantages)

    assert len(loss_worse) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    lw_loss, lw_reg, lw_kl, lw_ent, lw_lr, lw_clip_lr = loss_worse
    assert lw_kl.ndim == 1
    assert lw_kl.shape[0] == 1
    assert len(loss_better) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    lb_loss, lb_reg, lb_kl, lb_ent, lb_lr, lb_clip_lr = loss_better
    assert lb_kl.ndim == 1
    assert lb_kl.shape[0] == 1
    assert lw_kl > lb_kl
    assert lw_reg > lb_reg


@pytest.mark.unit_test
def test_clipped_ppo_loss_discrete_batch_time():
    batch_size = 3
    time_steps = 4
    num_actions = 2

    # actions taken, of shape (batch_size, time_steps)
    actions = mx.nd.array(((0, 1, 0, 0),
                           (1, 1, 0, 0),
                           (0, 0, 0, 0)))
    # advantages from taking action, of shape (batch_size, time_steps)
    advantages = mx.nd.array(((-2, 2, 1, 0),
                              (-1, 1, 0, 1),
                              (-1, 0, 1, 0)))
    # action probabilities, of shape (batch_size, num_actions)
    old_policy_probs = mx.nd.array((((0.8, 0.2), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                     ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                     ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))))
    new_policy_probs_worse = mx.nd.array((((0.9, 0.1), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                          ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                          ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))))
    new_policy_probs_better = mx.nd.array((((0.2, 0.8), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                           ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                                           ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))))

    # check lower loss for policy with better probabilities:
    # i.e. higher probability on high advantage actions, low probability on low advantage actions.
    loss_fn = ClippedPPOLossDiscrete(num_actions=num_actions,
                                     clip_likelihood_ratio_using_epsilon=0.2)
    loss_fn.initialize()

    clip_param_rescaler = mx.nd.array((1,))

    loss_worse = loss_fn(new_policy_probs_worse, actions, old_policy_probs, clip_param_rescaler, advantages)
    loss_better = loss_fn(new_policy_probs_better, actions, old_policy_probs, clip_param_rescaler, advantages)

    assert len(loss_worse) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    loss_worse_val = loss_worse[0]
    assert loss_worse_val.ndim == 1
    assert loss_worse_val.shape[0] == 1
    assert len(loss_better) == 6  # (LOSS, REGULARIZATION, KL, ENTROPY, LIKELIHOOD_RATIO, CLIPPED_LIKELIHOOD_RATIO)
    loss_better_val = loss_better[0]
    assert loss_better_val.ndim == 1
    assert loss_better_val.shape[0] == 1
    assert loss_worse_val > loss_better_val


@pytest.mark.unit_test
def test_clipped_ppo_loss_discrete_weight():
    actions = mx.nd.array((0, 1, 0))
    advantages = mx.nd.array((-2, 2, 1))
    old_policy_probs = mx.nd.array(((0.7, 0.3), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs = mx.nd.array(((0.9, 0.1), (0.2, 0.8), (0.4, 0.6)))

    clip_param_rescaler = mx.nd.array((1,))
    loss_fn = ClippedPPOLossDiscrete(num_actions=2,
                                     clip_likelihood_ratio_using_epsilon=0.2)
    loss_fn.initialize()
    loss = loss_fn(new_policy_probs, actions, old_policy_probs, clip_param_rescaler, advantages)
    loss_fn_weighted = ClippedPPOLossDiscrete(num_actions=2,
                                     clip_likelihood_ratio_using_epsilon=0.2,
                                     weight=0.5)
    loss_fn_weighted.initialize()
    loss_weighted = loss_fn_weighted(new_policy_probs, actions, old_policy_probs, clip_param_rescaler, advantages)
    assert loss[0] == loss_weighted[0] * 2


@pytest.mark.unit_test
def test_clipped_ppo_loss_discrete_hybridize():
    loss_fn = ClippedPPOLossDiscrete(num_actions=2,
                                     clip_likelihood_ratio_using_epsilon=0.2)
    loss_fn.initialize()
    loss_fn.hybridize()
    actions = mx.nd.array((0, 1, 0))
    advantages = mx.nd.array((-2, 2, 1))
    old_policy_probs = mx.nd.array(((0.7, 0.3), (0.2, 0.8), (0.4, 0.6)))
    new_policy_probs = mx.nd.array(((0.9, 0.1), (0.2, 0.8), (0.4, 0.6)))
    clip_param_rescaler = mx.nd.array((1,))

    loss = loss_fn(new_policy_probs, actions, old_policy_probs, clip_param_rescaler, advantages)
    assert loss[0] == mx.nd.array((-0.142857153,))


@pytest.mark.unit_test
def test_discrete_ppo_head():
    head = DiscretePPOHead(num_actions=2)
    head.initialize()
    middleware_data = mx.nd.random.uniform(shape=(10, 100))
    probs = head(middleware_data)
    assert probs.ndim == 2  # (batch_size, num_actions)
    assert probs.shape[0] == 10  # since batch_size is 10
    assert probs.shape[1] == 2  # since num_actions is 2


@pytest.mark.unit_test
def test_ppo_head():
    agent_parameters = ClippedPPOAgentParameters()
    num_actions = 5
    action_space = DiscreteActionSpace(num_actions=num_actions)
    spaces = SpacesDefinition(state=None, goal=None, action=action_space, reward=None)
    head = PPOHead(agent_parameters=agent_parameters,
                   spaces=spaces,
                   network_name="test_ppo_head")

    head.initialize()

    batch_size = 15
    middleware_data = mx.nd.random.uniform(shape=(batch_size, 100))
    probs = head(middleware_data)
    assert probs.ndim == 2  # (batch_size, num_actions)
    assert probs.shape[0] == batch_size
    assert probs.shape[1] == num_actions
