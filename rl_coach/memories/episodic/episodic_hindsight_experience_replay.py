#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from enum import Enum
from typing import Tuple, List

import numpy as np

from rl_coach.core_types import Episode, Transition
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters, \
    EpisodicExperienceReplay
from rl_coach.memories.non_episodic.experience_replay import MemoryGranularity
from rl_coach.spaces import GoalsSpace


class HindsightGoalSelectionMethod(Enum):
    Future = 0
    Final = 1
    Episode = 2
    Random = 3


class EpisodicHindsightExperienceReplayParameters(EpisodicExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.hindsight_transitions_per_regular_transition = None
        self.hindsight_goal_selection_method = None
        self.goals_space = None

    @property
    def path(self):
        return 'rl_coach.memories.episodic.episodic_hindsight_experience_replay:EpisodicHindsightExperienceReplay'


class EpisodicHindsightExperienceReplay(EpisodicExperienceReplay):
    """
    Implements Hindsight Experience Replay as described in the following paper: https://arxiv.org/pdf/1707.01495.pdf

    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int],
                 hindsight_transitions_per_regular_transition: int,
                 hindsight_goal_selection_method: HindsightGoalSelectionMethod,
                 goals_space: GoalsSpace):
        """
        :param max_size: The maximum size of the memory. should be defined in a granularity of Transitions
        :param hindsight_transitions_per_regular_transition: The number of hindsight artificial transitions to generate
                                                             for each actual transition
        :param hindsight_goal_selection_method: The method that will be used for generating the goals for the
                                                hindsight transitions. Should be one of HindsightGoalSelectionMethod
        :param goals_space: A GoalsSpace which defines the base properties of the goals space
        """
        super().__init__(max_size)

        self.hindsight_transitions_per_regular_transition = hindsight_transitions_per_regular_transition
        self.hindsight_goal_selection_method = hindsight_goal_selection_method
        self.goals_space = goals_space
        self.last_episode_start_idx = 0

    def _sample_goal(self, episode_transitions: List, transition_index: int):
        """
        Sample a single goal state according to the sampling method
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_index: the transition to start sampling from
        :return: a goal corresponding to the sampled state
        """
        if self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Future:
            # states that were observed in the same episode after the transition that is being replayed
            selected_transition = np.random.choice(episode_transitions[transition_index+1:])
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Final:
            # the final state in the episode
            selected_transition = episode_transitions[-1]
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Episode:
            # a random state from the episode
            selected_transition = np.random.choice(episode_transitions)
        elif self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Random:
            # a random state from the entire replay buffer
            selected_transition = np.random.choice(self.transitions)
        else:
            raise ValueError("Invalid goal selection method was used for the hindsight goal selection")
        return self.goals_space.goal_from_state(selected_transition.state)

    def _sample_goals(self, episode_transitions: List, transition_index: int):
        """
        Sample a batch of goal states according to the sampling method
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_index: the transition to start sampling from
        :return: a goal corresponding to the sampled state
        """
        return [
            self._sample_goal(episode_transitions, transition_index)
            for _ in range(self.hindsight_transitions_per_regular_transition)
        ]

    def store_episode(self, episode: Episode, lock: bool=True) -> None:
        # generate hindsight transitions only when an episode is finished
        last_episode_transitions = copy.copy(episode.transitions)

        # cannot create a future hindsight goal in the last transition of an episode
        if self.hindsight_goal_selection_method == HindsightGoalSelectionMethod.Future:
            relevant_base_transitions = last_episode_transitions[:-1]
        else:
            relevant_base_transitions = last_episode_transitions

        # for each transition in the last episode, create a set of hindsight transitions
        for transition_index, transition in enumerate(relevant_base_transitions):
            sampled_goals = self._sample_goals(last_episode_transitions, transition_index)
            for goal in sampled_goals:
                hindsight_transition = copy.copy(transition)

                if hindsight_transition.state['desired_goal'].shape != goal.shape:
                    raise ValueError((
                        'goal shape {goal_shape} already in transition is '
                        'different than the one sampled as a hindsight goal '
                        '{hindsight_goal_shape}.'
                    ).format(
                        goal_shape=hindsight_transition.state['desired_goal'].shape,
                        hindsight_goal_shape=goal.shape,
                    ))

                # update the goal in the transition
                hindsight_transition.state['desired_goal'] = goal
                hindsight_transition.next_state['desired_goal'] = goal

                # update the reward and terminal signal according to the goal
                hindsight_transition.reward, hindsight_transition.game_over = \
                    self.goals_space.get_reward_for_goal_and_state(goal, hindsight_transition.next_state)

                hindsight_transition.total_return = None
                episode.insert(hindsight_transition)

        super().store_episode(episode)

    def store(self, transition: Transition):
        raise ValueError("An episodic HER cannot store a single transition. Only full episodes are to be stored.")
