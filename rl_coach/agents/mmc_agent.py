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

from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters


class MixedMonteCarloAlgorithmParameters(DQNAlgorithmParameters):
    """
    :param monte_carlo_mixing_rate: (float)
        The mixing rate is used for setting the amount of monte carlo estimate (full return) that will be mixes into
        the single-step bootstrapped targets.
    """
    def __init__(self):
        super().__init__()
        self.monte_carlo_mixing_rate = 0.1


class MixedMonteCarloAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = MixedMonteCarloAlgorithmParameters()
        self.memory = EpisodicExperienceReplayParameters()

    @property
    def path(self):
        return 'rl_coach.agents.mmc_agent:MixedMonteCarloAgent'


class MixedMonteCarloAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.mixing_rate = agent_parameters.algorithm.monte_carlo_mixing_rate

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # for the 1-step, we use the double-dqn target. hence actions are taken greedily according to the online network
        selected_actions = np.argmax(self.networks['main'].online_network.predict(batch.next_states(network_keys)), 1)

        # TD_targets are initialized with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        total_returns = batch.n_step_discounted_rewards()

        for i in range(self.ap.network_wrappers['main'].batch_size):
            one_step_target = batch.rewards()[i] + \
                              (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * \
                              q_st_plus_1[i][selected_actions[i]]
            monte_carlo_target = total_returns[i]
            TD_targets[i, batch.actions()[i]] = (1 - self.mixing_rate) * one_step_target + \
                                                self.mixing_rate * monte_carlo_target

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
