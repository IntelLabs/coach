# nasty hack to deal with issue #46
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

from rl_coach.core_types import Transition
from rl_coach.memories.episodic.single_episode_buffer import SingleEpisodeBuffer


@pytest.fixture()
def buffer():
    return SingleEpisodeBuffer()


@pytest.mark.unit_test
def test_store_and_get(buffer: SingleEpisodeBuffer):
    # store single non terminal transition
    transition = Transition(state={"observation": np.array([1, 2, 3])}, action=1, reward=1, game_over=False)
    buffer.store(transition)
    assert buffer.length() == 1
    assert buffer.num_complete_episodes() == 0
    assert buffer.num_transitions_in_complete_episodes() == 0
    assert buffer.num_transitions() == 1

    # get the single stored transition
    episode = buffer.get(0)
    assert episode.length() == 1
    assert episode.get_first_transition() is transition    # check addresses are the same
    assert episode.get_last_transition() is transition   # check addresses are the same

    # store single terminal transition
    transition = Transition(state={"observation": np.array([1, 2, 3])}, action=1, reward=1, game_over=True)
    buffer.store(transition)
    assert buffer.length() == 1
    assert buffer.num_complete_episodes() == 1
    assert buffer.num_transitions_in_complete_episodes() == 2

    # check that the episode is valid
    episode = buffer.get(0)
    assert episode.length() == 2
    assert episode.get_transition(0).n_step_discounted_rewards == 1 + 0.99
    assert episode.get_transition(1).n_step_discounted_rewards == 1
    assert buffer.mean_reward() == 1

    # only one episode in the replay buffer
    episode = buffer.get(1)
    assert episode is None

    # adding transitions after the first episode was closed
    transition = Transition(state={"observation": np.array([1, 2, 3])}, action=1, reward=0, game_over=False)
    buffer.store(transition)
    assert buffer.length() == 1
    assert buffer.num_complete_episodes() == 0
    assert buffer.num_transitions_in_complete_episodes() == 0

    # still only one episode
    assert buffer.get(1) is None
    assert buffer.mean_reward() == 0


@pytest.mark.unit_test
def test_clean(buffer: SingleEpisodeBuffer):
    # add several transitions and then clean the buffer
    transition = Transition(state={"observation": np.array([1, 2, 3])}, action=1, reward=1, game_over=False)
    for i in range(10):
        buffer.store(transition)
    assert buffer.num_transitions() == 10
    buffer.clean()
    assert buffer.num_transitions() == 0

    # add more transitions after the clean and make sure they were really cleaned
    transition = Transition(state={"observation": np.array([1, 2, 3])}, action=1, reward=1, game_over=True)
    buffer.store(transition)
    assert buffer.num_transitions() == 1
    assert buffer.num_transitions_in_complete_episodes() == 1
    assert buffer.num_complete_episodes() == 1
    for i in range(10):
        assert buffer.sample(1)[0] is transition
