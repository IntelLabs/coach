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

from typing import Union

import numpy as np

from rl_coach.agents.ddpg_agent import DDPGAgent, DDPGAgentParameters, DDPGAlgorithmParameters
from rl_coach.core_types import RunPhase
from rl_coach.spaces import SpacesDefinition


class HACDDPGAlgorithmParameters(DDPGAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.time_limit = 40
        self.sub_goal_testing_rate = 0.5


class HACDDPGAgentParameters(DDPGAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = HACDDPGAlgorithmParameters()

    @property
    def path(self):
        return 'rl_coach.agents.hac_ddpg_agent:HACDDPGAgent'


# Hierarchical Actor Critic Generating Subgoals DDPG Agent - https://arxiv.org/pdf/1712.00948.pdf
class HACDDPGAgent(DDPGAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.sub_goal_testing_rate = self.ap.algorithm.sub_goal_testing_rate
        self.graph_manager = None

    def choose_action(self, curr_state):
        # top level decides, for each of his generated sub-goals, for all the layers beneath him if this is a sub-goal
        # testing phase

        graph_manager = self.parent_level_manager.parent_graph_manager
        if self.ap.is_a_highest_level_agent:
            graph_manager.should_test_current_sub_goal = np.random.rand() < self.sub_goal_testing_rate

        if self.phase == RunPhase.TRAIN:
            if graph_manager.should_test_current_sub_goal:
                self.exploration_policy.change_phase(RunPhase.TEST)
            else:
                self.exploration_policy.change_phase(self.phase)

        action_info = super().choose_action(curr_state)
        return action_info

    def update_transition_before_adding_to_replay_buffer(self, transition):
        graph_manager = self.parent_level_manager.parent_graph_manager

        # deal with goals given from a higher level agent
        if not self.ap.is_a_highest_level_agent:
            transition.state['desired_goal'] = self.current_hrl_goal
            transition.next_state['desired_goal'] = self.current_hrl_goal
            # TODO: allow setting goals which are not part of the state. e.g. state-embedding using get_prediction
            self.distance_from_goal.add_sample(self.spaces.goal.distance_from_goal(
                self.current_hrl_goal, transition.next_state))
            goal_reward, sub_goal_reached = self.spaces.goal.get_reward_for_goal_and_state(
                self.current_hrl_goal, transition.next_state)
            transition.reward = goal_reward
            transition.game_over = transition.game_over or sub_goal_reached

        # each level tests its own generated sub goals
        if not self.ap.is_a_lowest_level_agent and graph_manager.should_test_current_sub_goal:
            #TODO-fixme
            # _, sub_goal_reached = self.parent_level_manager.environment.agents['agent_1'].spaces.goal.\
            # get_reward_for_goal_and_state(transition.action, transition.next_state)

            _, sub_goal_reached = self.spaces.goal.get_reward_for_goal_and_state(
                transition.action, transition.next_state)

            sub_goal_is_missed = not sub_goal_reached

            if sub_goal_is_missed:
                    transition.reward = -self.ap.algorithm.time_limit
        return transition

    def set_environment_parameters(self, spaces: SpacesDefinition):
        super().set_environment_parameters(spaces)

        if self.ap.is_a_highest_level_agent:
            # the rest of the levels already have an in_action_space set to be of type GoalsSpace, thus they will have
            # their GoalsSpace set to the in_action_space in agent.set_environment_parameters()
            self.spaces.goal = self.spaces.action
            self.spaces.goal.set_target_space(self.spaces.state[self.spaces.goal.goal_name])

        if not self.ap.is_a_highest_level_agent:
            self.spaces.reward.reward_success_threshold = self.spaces.goal.reward_type.goal_reaching_reward
