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

from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.core_types import EnvironmentSteps
from rl_coach.schedules import LinearSchedule


class DDQNAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(30000)
        self.exploration.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
        self.exploration.evaluation_epsilon = 0.001

    @property
    def path(self):
        return 'rl_coach.agents.ddqn_agent:DDQNAgent'


# Double DQN - https://arxiv.org/abs/1509.06461
class DDQNAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        selected_actions = np.argmax(self.networks['main'].online_network.predict(batch.next_states(network_keys)), 1)
        q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        # initialize with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        TD_errors = []
        for i in range(self.ap.network_wrappers['main'].batch_size):
            new_target = batch.rewards()[i] + \
                         (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * q_st_plus_1[i][selected_actions[i]]
            TD_errors.append(np.abs(new_target - TD_targets[i, batch.actions()[i]]))
            TD_targets[i, batch.actions()[i]] = new_target

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(TD_errors, batch)

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets,
                                                               importance_weights=importance_weights)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
