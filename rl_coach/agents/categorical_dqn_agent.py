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
from rl_coach.agents.dqn_agent import DQNNetworkParameters, DQNAlgorithmParameters, DQNAgentParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.architectures.head_parameters import CategoricalQHeadParameters
from rl_coach.core_types import StateType
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplay
from rl_coach.schedules import LinearSchedule


class CategoricalDQNNetworkParameters(DQNNetworkParameters):
    def __init__(self):
        super().__init__()
        self.heads_parameters = [CategoricalQHeadParameters()]


class CategoricalDQNAlgorithmParameters(DQNAlgorithmParameters):
    """
    :param v_min: (float)
        The minimal value that will be represented in the network output for predicting the Q value.
        Corresponds to :math:`v_{min}` in the paper.

    :param v_max: (float)
        The maximum value that will be represented in the network output for predicting the Q value.
        Corresponds to :math:`v_{max}` in the paper.

    :param atoms: (int)
        The number of atoms that will be used to discretize the range between v_min and v_max.
        For the C51 algorithm described in the paper, the number of atoms is 51.
    """
    def __init__(self):
        super().__init__()
        self.v_min = -10.0
        self.v_max = 10.0
        self.atoms = 51


class CategoricalDQNExplorationParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
        self.evaluation_epsilon = 0.001


class CategoricalDQNAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = CategoricalDQNAlgorithmParameters()
        self.exploration = CategoricalDQNExplorationParameters()
        self.network_wrappers = {"main": CategoricalDQNNetworkParameters()}

    @property
    def path(self):
        return 'rl_coach.agents.categorical_dqn_agent:CategoricalDQNAgent'


# Categorical Deep Q Network - https://arxiv.org/pdf/1707.06887.pdf
class CategoricalDQNAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.z_values = np.linspace(self.ap.algorithm.v_min, self.ap.algorithm.v_max, self.ap.algorithm.atoms)

    def distribution_prediction_to_q_values(self, prediction):
        return np.dot(prediction, self.z_values)

    # prediction's format is (batch,actions,atoms)
    def get_all_q_values_for_states(self, states: StateType):
        if self.exploration_policy.requires_action_values():
            prediction = self.get_prediction(states)
            q_values = self.distribution_prediction_to_q_values(prediction)
        else:
            q_values = None
        return q_values

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # for the action we actually took, the error is calculated by the atoms distribution
        # for all other actions, the error is 0
        distributional_q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        # select the optimal actions for the next state
        target_actions = np.argmax(self.distribution_prediction_to_q_values(distributional_q_st_plus_1), axis=1)
        m = np.zeros((self.ap.network_wrappers['main'].batch_size, self.z_values.size))

        batches = np.arange(self.ap.network_wrappers['main'].batch_size)

        # an alternative to the for loop. 3.7x perf improvement vs. the same code done with for looping.
        # only 10% speedup overall - leaving commented out as the code is not as clear.

        # tzj_ = np.fmax(np.fmin(batch.rewards() + (1.0 - batch.game_overs()) * self.ap.algorithm.discount *
        #                        np.transpose(np.repeat(self.z_values[np.newaxis, :], batch.size, axis=0), (1, 0)),
        #                     self.z_values[-1]),
        #             self.z_values[0])
        #
        # bj_ = (tzj_ - self.z_values[0]) / (self.z_values[1] - self.z_values[0])
        # u_ = (np.ceil(bj_)).astype(int)
        # l_ = (np.floor(bj_)).astype(int)
        # m_ = np.zeros((self.ap.network_wrappers['main'].batch_size, self.z_values.size))
        # np.add.at(m_, [batches, l_],
        #           np.transpose(distributional_q_st_plus_1[batches, target_actions], (1, 0)) * (u_ - bj_))
        # np.add.at(m_, [batches, u_],
        #           np.transpose(distributional_q_st_plus_1[batches, target_actions], (1, 0)) * (bj_ - l_))

        for j in range(self.z_values.size):
            tzj = np.fmax(np.fmin(batch.rewards() +
                                  (1.0 - batch.game_overs()) * self.ap.algorithm.discount * self.z_values[j],
                                  self.z_values[-1]),
                          self.z_values[0])
            bj = (tzj - self.z_values[0])/(self.z_values[1] - self.z_values[0])
            u = (np.ceil(bj)).astype(int)
            l = (np.floor(bj)).astype(int)
            m[batches, l] += (distributional_q_st_plus_1[batches, target_actions, j] * (u - bj))
            m[batches, u] += (distributional_q_st_plus_1[batches, target_actions, j] * (bj - l))

        # total_loss = cross entropy between actual result above and predicted result for the given action
        # only update the action that we have actually done in this transition
        TD_targets[batches, batch.actions()] = m

        # update errors in prioritized replay buffer
        importance_weights = batch.info('weight') if isinstance(self.memory, PrioritizedExperienceReplay) else None

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets,
                                                               importance_weights=importance_weights)

        total_loss, losses, unclipped_grads = result[:3]

        # TODO: fix this spaghetti code
        if isinstance(self.memory, PrioritizedExperienceReplay):
            errors = losses[0][np.arange(batch.size), batch.actions()]
            self.call_memory('update_priorities', (batch.info('idx'), errors))

        return total_loss, losses, unclipped_grads

