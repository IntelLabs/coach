import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.exploration_policies.ou_process import OUProcess
from rl_coach.core_types import RunPhase
import numpy as np


@pytest.mark.unit_test
def test_init():
    # discrete control
    action_space = DiscreteActionSpace(3)

    # OU process doesn't work for discrete controls
    with pytest.raises(ValueError):
        policy = OUProcess(action_space, mu=0, theta=0.1, sigma=0.2, dt=0.01)


@pytest.mark.unit_test
def test_get_action():
    action_space = BoxActionSpace(np.array([10]), -1, 1)
    policy = OUProcess(action_space, mu=0, theta=0.1, sigma=0.2, dt=0.01)

    # make sure no noise is added in the testing phase
    policy.change_phase(RunPhase.TEST)
    assert np.all(policy.get_action(np.zeros((10,))) == np.zeros((10,)))
    rand_action = np.random.rand(10)
    assert np.all(policy.get_action(rand_action) == rand_action)

    # make sure the noise added in the training phase matches the golden
    policy.change_phase(RunPhase.TRAIN)
    np.random.seed(0)
    targets = [
        [0.03528105, 0.00800314, 0.01957476, 0.04481786, 0.03735116, - 0.01954556, 0.01900177, - 0.00302714, - 0.00206438, 0.00821197],
        [0.03812664, 0.03708061, 0.03477594, 0.04720655, 0.04619107, - 0.01285253, 0.04886435, - 0.00712728, 0.00419904, - 0.00887816],
        [-0.01297129, 0.0501159, 0.05202989, 0.03231604, 0.09153997, - 0.04192699, 0.04973065, - 0.01086383, 0.03485043, 0.0205179],
        [-0.00985937, 0.05762904, 0.03422214, - 0.00733221, 0.08449019, - 0.03875808, 0.07428674, 0.01319463, 0.02706904, 0.01445132],
        [-3.08205658e-02, 2.91710492e-02, 6.25166679e-05, 3.16906342e-02, 7.42126579e-02, - 4.74808080e-02, 4.91565431e-02, 2.87312413e-02, - 5.23598615e-03, 1.01820670e-02],
        [-0.04869908, 0.03687993, - 0.01015365, 0.0080463, 0.0735748, -0.03886669, 0.05043773, 0.03475195, - 0.01791719, 0.00291706],
        [-0.06209959, 0.02965198, - 0.02640642, - 0.0264874, 0.07704975, - 0.04686344, 0.01778333, 0.04397284, - 0.03604524, 0.00395305],
        [-0.04745568, 0.03220199, - 0.003592, -0.05115743, 0.08501953, - 0.06051278, 0.0003496, 0.03235188, - 0.04224025, 0.00507241],
        [-0.07071122, 0.05018632, 0.00572484, - 0.08183114, 0.11469956, - 0.02253448, 0.02392484, 0.02872103, - 0.06361306, 0.02615637],
        [-0.07870404, 0.07458503, 0.00988462, - 0.06221653, 0.12171218, - 0.00838049, 0.02411092, 0.06440972, - 0.0610112, 0.03417],
        [-0.04096233, 0.04755527, - 0.01553497, - 0.04276638, 0.098128, 0.03050032, 0.01581443, 0.04939621, - 0.02249135, 0.06374613],
        [-0.00357018, 0.06562861, - 0.03274395, - 0.00452232, 0.09266981, 0.04651895, 0.03474365, 0.04624661, - 0.01018727, 0.08212651],
    ]
    for i in range(10):
        current_noise = policy.get_action(np.zeros((10,)))
        assert np.all(np.abs(current_noise - targets[i]) < 1e-7)

    # get some statistics. check very roughly that the mean acts according to the definition of the policy

    # mean of 0
    vals = []
    for i in range(50000):
        current_noise = policy.get_action(np.zeros((10,)))
        vals.append(current_noise)
    assert np.all(np.abs(np.mean(vals, axis=0)) < 1)

    # mean of 10
    policy = OUProcess(action_space, mu=10, theta=0.1, sigma=0.2, dt=0.01)
    policy.change_phase(RunPhase.TRAIN)
    vals = []
    for i in range(50000):
        current_noise = policy.get_action(np.zeros((10,)))
        vals.append(current_noise)
    assert np.all(np.abs(np.mean(vals, axis=0) - 10) < 1)

    # plot the noise values - only used for understanding how the noise actually looks
    # import matplotlib.pyplot as plt
    # vals = np.array(vals)
    # for i in range(10):
    #     plt.plot(list(range(10000)), vals[:, i])
    #     plt.plot(list(range(10000)), vals[:, i])
    #     plt.plot(list(range(10000)), vals[:, i])
    # plt.show()


if __name__ == "__main__":
    test_get_action()
