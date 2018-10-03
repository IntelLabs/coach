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

from typing import Tuple

from rl_coach.core_types import Episode, Transition
from rl_coach.memories.episodic.episodic_hindsight_experience_replay import HindsightGoalSelectionMethod, \
    EpisodicHindsightExperienceReplay, EpisodicHindsightExperienceReplayParameters
from rl_coach.memories.non_episodic.experience_replay import MemoryGranularity
from rl_coach.spaces import GoalsSpace


class EpisodicHRLHindsightExperienceReplayParameters(EpisodicHindsightExperienceReplayParameters):
    def __init__(self):
        super().__init__()

    @property
    def path(self):
        return 'rl_coach.memories.episodic.episodic_hrl_hindsight_experience_replay:EpisodicHRLHindsightExperienceReplay'


class EpisodicHRLHindsightExperienceReplay(EpisodicHindsightExperienceReplay):
    """
    Implements HRL Hindsight Experience Replay as described in the following paper:  https://arxiv.org/abs/1805.08180

    This is the memory you should use if you want a shared hindsight experience replay buffer between multiple workers
    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int],
                 hindsight_transitions_per_regular_transition: int,
                 hindsight_goal_selection_method: HindsightGoalSelectionMethod,
                 goals_space: GoalsSpace,
                 ):
        """
        :param max_size: The maximum size of the memory. should be defined in a granularity of Transitions
        :param hindsight_transitions_per_regular_transition: The number of hindsight artificial transitions to generate
                                                             for each actual transition
        :param hindsight_goal_selection_method: The method that will be used for generating the goals for the
                                                hindsight transitions. Should be one of HindsightGoalSelectionMethod
        :param goals_space: A GoalsSpace  which defines the properties of the goals
        :param do_action_hindsight: Replace the action (sub-goal) given to a lower layer, with the actual achieved goal
        """
        super().__init__(max_size, hindsight_transitions_per_regular_transition, hindsight_goal_selection_method,
                         goals_space)

    def store_episode(self, episode: Episode, lock: bool=True) -> None:
        # for a layer producing sub-goals, we will replace in hindsight the action (sub-goal) given to the lower
        # level with the actual achieved goal. the achieved goal (and observation) seen is assumed to be the same
        # for all levels - we can use this level's achieved goal instead of the lower level's one

        # Calling super.store() so that in case a memory backend is used, the memory backend can store this episode.
        super().store_episode(episode)

        for transition in episode.transitions:
            new_achieved_goal = transition.next_state[self.goals_space.goal_name]
            transition.action = new_achieved_goal

        super().store_episode(episode)

    def store(self, transition: Transition):
        raise ValueError("An episodic HER cannot store a single transition. Only full episodes are to be stored.")
