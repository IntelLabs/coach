import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, MultiSelectActionSpace, ObservationSpace, AgentSelection, VectorObservationSpace, AttentionActionSpace
import numpy as np


@pytest.mark.unit_test
def test_discrete():
    action_space = DiscreteActionSpace(3, ["zero", "one", "two"])
    assert action_space.shape == 1
    for i in range(100):
        assert 3 > action_space.sample() >= 0
    action_info = action_space.sample_with_info()
    assert action_info.action_probability == 1. / 3
    assert action_space.high == 2
    assert action_space.low == 0

    # list descriptions
    assert action_space.get_description(1) == "one"

    # dict descriptions
    action_space = DiscreteActionSpace(3, {1: "one", 2: "two", 0: "zero"})
    assert action_space.get_description(0) == "zero"

    # no descriptions
    action_space = DiscreteActionSpace(3)
    assert action_space.get_description(0) == "0"

    # descriptions for invalid action
    with pytest.raises(ValueError):
        assert action_space.get_description(3) == "0"


@pytest.mark.unit_test
def test_box():
    # simple action space
    action_space = BoxActionSpace(4, -5, 5, ["a", "b", "c", "d"])
    for i in range(100):
        sample = action_space.sample()
        assert np.all(-5 <= sample) and np.all(sample <= 5)
        assert sample.shape == (4,)
        assert sample.dtype == float

    # test clipping
    clipped_action = action_space.clip_action_to_space(np.array([-10, 10, 2, 5]))
    assert np.all(clipped_action == np.array([-5, 5, 2, 5]))

    # more complex high and low definition
    action_space = BoxActionSpace(4, np.array([-5, -1, -0.5, 0]), np.array([1, 2, 4, 5]), ["a", "b", "c", "d"])
    for i in range(100):
        sample = action_space.sample()
        assert np.all(np.array([-5, -1, -0.5, 0]) <= sample) and np.all(sample <= np.array([1, 2, 4, 5]))
        assert sample.shape == (4,)
        assert sample.dtype == float

    # test clipping
    clipped_action = action_space.clip_action_to_space(np.array([-10, 10, 2, 5]))
    assert np.all(clipped_action == np.array([-5, 2, 2, 5]))

    # mixed high and low definition
    action_space = BoxActionSpace(4, np.array([-5, -1, -0.5, 0]), 5, ["a", "b", "c", "d"])
    for i in range(100):
        sample = action_space.sample()
        assert np.all(np.array([-5, -1, -0.5, 0]) <= sample) and np.all(sample <= 5)
        assert sample.shape == (4,)
        assert sample.dtype == float

    # test clipping
    clipped_action = action_space.clip_action_to_space(np.array([-10, 10, 2, 5]))
    assert np.all(clipped_action == np.array([-5, 5, 2, 5]))

    # invalid bounds
    with pytest.raises(ValueError):
        action_space = BoxActionSpace(4, np.array([-5, -1, -0.5, 0]), -1, ["a", "b", "c", "d"])

    # TODO: test descriptions


@pytest.mark.unit_test
def test_multiselect():
    action_space = MultiSelectActionSpace(4, 2, ["a", "b", "c", "d"])
    for i in range(100):
        action = action_space.sample()
        assert action.shape == (4,)
        assert np.sum(action) <= 2

    # check that descriptions of multiple actions are working
    description = action_space.get_description(np.array([1, 0, 1, 0]))
    assert description == "a + c"

    description = action_space.get_description(np.array([0, 0, 0, 0]))
    assert description == "no-op"


@pytest.mark.unit_test
def test_attention():
    low = np.array([-1, -2, -3, -4])
    high = np.array([1, 2, 3, 4])
    action_space = AttentionActionSpace(4, low=low, high=high)
    for i in range(100):
        action = action_space.sample()
        assert len(action) == 2
        assert action[0].shape == (4,)
        assert action[1].shape == (4,)
        assert np.all(action[0] <= action[1])
        assert np.all(action[0] >= low)
        assert np.all(action[1] < high)


@pytest.mark.unit_test
def test_goal():
    # TODO: test goal action space
    pass


@pytest.mark.unit_test
def test_agent_selection():
    action_space = AgentSelection(10)

    assert action_space.shape == 1
    assert action_space.high == 9
    assert action_space.low == 0
    with pytest.raises(ValueError):
        assert action_space.get_description(10)
    assert action_space.get_description(0) == "0"


@pytest.mark.unit_test
def test_observation_space():
    observation_space = ObservationSpace(np.array([1, 10]), -10, 10)

    # testing that val_matches_space_definition works
    assert observation_space.val_matches_space_definition(np.ones([1, 10]))
    assert not observation_space.val_matches_space_definition(np.ones([2, 10]))
    assert not observation_space.val_matches_space_definition(np.ones([1, 10]) * 100)
    assert not observation_space.val_matches_space_definition(np.ones([1, 1, 10]))

    # is_point_in_space_shape
    assert observation_space.is_point_in_space_shape(np.array([0, 9]))
    assert observation_space.is_point_in_space_shape(np.array([0, 0]))
    assert not observation_space.is_point_in_space_shape(np.array([1, 8]))
    assert not observation_space.is_point_in_space_shape(np.array([0, 10]))
    assert not observation_space.is_point_in_space_shape(np.array([-1, 6]))


@pytest.mark.unit_test
def test_image_observation_space():
    # TODO: test image observation space
    pass


@pytest.mark.unit_test
def test_measurements_observation_space():
    # empty measurements space
    measurements_space = VectorObservationSpace(0)

    # vector space
    measurements_space = VectorObservationSpace(3, measurements_names=['a', 'b', 'c'])


@pytest.mark.unit_test
def test_reward_space():
    # TODO: test reward space
    pass


# def test_discrete_to_linspace_action_space_map():
#     box = BoxActionSpace(2, np.array([0, 0]), np.array([10, 10]))
#     linspace = BoxDiscretization(box, [5, 3])
#     assert np.all(linspace.actions == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))
#     assert np.all(linspace.target_actions ==
#                   np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0],
#                             [2.5, 0.0], [2.5, 5.0], [2.5, 10.0],
#                             [5.0, 0.0], [5.0, 5.0], [5.0, 10.0],
#                             [7.5, 0.0], [7.5, 5.0], [7.5, 10.0],
#                             [10.0, 0.0], [10.0, 5.0], [10.0, 10.0]]))
#
#
# def test_discrete_to_attention_action_space_map():
#     attention = AttentionActionSpace(2, np.array([0, 0]), np.array([10, 10]))
#     linspace = AttentionDiscretization(attention, 2)
#     assert np.all(linspace.actions == np.array([0, 1, 2, 3]))
#     assert np.all(linspace.target_actions ==
#                   np.array(
#                       [[[0., 0.], [5., 5.]],
#                       [[0., 5.], [5., 10.]],
#                       [[5., 0.], [10., 5.]],
#                       [[5., 5.], [10., 10.]]])
#                   )


if __name__ == "__main__":
    test_observation_space()
    test_discrete_to_linspace_action_space_map()
    test_discrete_to_attention_action_space_map()
