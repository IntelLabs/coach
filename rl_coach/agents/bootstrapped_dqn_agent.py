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

from rl_coach.agents.dqn_agent import DQNNetworkParameters, DQNAlgorithmParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.base_parameters import AgentParameters
from rl_coach.exploration_policies.bootstrapped import BootstrappedParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters


class BootstrappedDQNNetworkParameters(DQNNetworkParameters):
    def __init__(self):
        super().__init__()
        self.num_output_head_copies = 10
        self.rescale_gradient_from_head_by_factor = [1.0/self.num_output_head_copies]*self.num_output_head_copies


class BootstrappedDQNAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DQNAlgorithmParameters(),
                         exploration=BootstrappedParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={"main": BootstrappedDQNNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.bootstrapped_dqn_agent:BootstrappedDQNAgent'


# Bootstrapped DQN - https://arxiv.org/pdf/1602.04621.pdf
class BootstrappedDQNAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def reset_internal_state(self):
        super().reset_internal_state()
        self.exploration_policy.select_head()

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        next_states_online_values = self.networks['main'].online_network.predict(batch.next_states(network_keys))
        result = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])
        q_st_plus_1 = result[:self.ap.exploration.architecture_num_q_heads]
        TD_targets = result[self.ap.exploration.architecture_num_q_heads:]

        # initialize with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        for i in range(self.ap.network_wrappers['main'].batch_size):
            mask = batch[i].info['mask']
            for head_idx in range(self.ap.exploration.architecture_num_q_heads):
                if mask[head_idx] == 1:
                    selected_action = np.argmax(next_states_online_values[head_idx][i], 0)
                    TD_targets[head_idx][i, batch.actions()[i]] = \
                        batch.rewards()[i] + (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount \
                                     * q_st_plus_1[head_idx][i][selected_action]

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

    def observe(self, env_response):
        mask = np.random.binomial(1, self.ap.exploration.bootstrapped_data_sharing_probability,
                                  self.ap.exploration.architecture_num_q_heads)
        env_response.info['mask'] = mask
        return super().observe(env_response)
