# nasty hack to deal with issue #46
import os
import sys

from rl_coach.memories.episodic.episodic_hindsight_experience_replay import EpisodicHindsightExperienceReplayParameters
from rl_coach.spaces import GoalsSpace, ReachingGoal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# print(sys.path)

import pytest
import numpy as np

from rl_coach.core_types import Transition, Episode
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.episodic.episodic_hindsight_experience_replay import EpisodicHindsightExperienceReplay, \
     HindsightGoalSelectionMethod


#TODO: change from defining a new class to creating an instance from the parameters
class Parameters(EpisodicHindsightExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 100)
        self.hindsight_transitions_per_regular_transition = 4
        self.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
        self.goals_space = GoalsSpace(goal_name='observation',
                                      reward_type=ReachingGoal(distance_from_goal_threshold=0.1),
                                      distance_metric=GoalsSpace.DistanceMetric.Euclidean)


@pytest.fixture
def episode():
    episode = []
    for i in range(10):
        episode.append(Transition(
            state={'observation': np.array([i]), 'desired_goal': np.array([i]), 'achieved_goal': np.array([i])},
            action=i,
        ))
    return episode


@pytest.fixture
def her():
    params = Parameters().__dict__

    import inspect
    args = set(inspect.getfullargspec(EpisodicHindsightExperienceReplay.__init__).args).intersection(params)
    params = {k: params[k] for k in args}

    return EpisodicHindsightExperienceReplay(**params)


@pytest.mark.unit_test
def test_sample_goal(her, episode):
    assert her._sample_goal(episode, 8) == 9


@pytest.mark.unit_test
def test_sample_goal_range(her, episode):
    unseen_goals = set(range(1, 9))
    for _ in range(500):
        unseen_goals -= set([int(her._sample_goal(episode, 0))])
        if not unseen_goals:
            return

    assert unseen_goals == set()


@pytest.mark.unit_test
def test_update_episode(her):
    episode = Episode()
    for i in range(10):
        episode.insert(Transition(
            state={'observation': np.array([i]), 'desired_goal': np.array([i+1]), 'achieved_goal': np.array([i+1])},
            action=i,
            game_over=i == 9,
            reward=0 if i == 9 else -1,
        ))

    her.store_episode(episode)
    # print('her._num_transitions', her._num_transitions)

    # 10 original transitions, and 9 transitions * 4 hindsight episodes
    assert her.num_transitions() == 10 + (4 * 9)

    # make sure that the goal state was never sampled from the past
    for transition in her.transitions:
        assert transition.state['desired_goal'] > transition.state['observation']
        assert transition.next_state['desired_goal'] >= transition.next_state['observation']

        if transition.reward == 0:
            assert transition.game_over
        else:
            assert not transition.game_over

test_update_episode(her())