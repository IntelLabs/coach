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


class PALAlgorithmParameters(DQNAlgorithmParameters):
    """
    :param pal_alpha: (float)
        A factor that weights the amount by which the advantage learning update will be taken into account.

    :param persistent_advantage_learning: (bool)
        If set to True, the persistent mode of advantage learning will be used, which encourages the agent to take
        the same actions one after the other instead of changing actions.

    :param monte_carlo_mixing_rate: (float)
        The amount of monte carlo values to mix into the targets of the network. The monte carlo values are just the
        total discounted returns, and they can help reduce the time it takes for the network to update to the newly
        seen values, since it is not based on bootstrapping the current network values.
    """
    def __init__(self):
        super().__init__()
        self.pal_alpha = 0.9
        self.persistent_advantage_learning = False
        self.monte_carlo_mixing_rate = 0.1


class PALAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = PALAlgorithmParameters()
        self.memory = EpisodicExperienceReplayParameters()

    @property
    def path(self):
        return 'rl_coach.agents.pal_agent:PALAgent'


# Persistent Advantage Learning - https://arxiv.org/pdf/1512.04860.pdf
class PALAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.alpha = agent_parameters.algorithm.pal_alpha
        self.persistent = agent_parameters.algorithm.persistent_advantage_learning
        self.monte_carlo_mixing_rate = agent_parameters.algorithm.monte_carlo_mixing_rate

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # next state values
        q_st_plus_1_target, q_st_plus_1_online = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.next_states(network_keys))
        ])
        selected_actions = np.argmax(q_st_plus_1_online, 1)
        v_st_plus_1_target = np.max(q_st_plus_1_target, 1)

        # current state values
        q_st_target, q_st_online = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])
        v_st_target = np.max(q_st_target, 1)

        # calculate TD error
        TD_targets = np.copy(q_st_online)
        total_returns = batch.n_step_discounted_rewards()
        for i in range(self.ap.network_wrappers['main'].batch_size):
            TD_targets[i, batch.actions()[i]] = batch.rewards()[i] + \
                                        (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * \
                                                     q_st_plus_1_target[i][selected_actions[i]]
            advantage_learning_update = v_st_target[i] - q_st_target[i, batch.actions()[i]]
            next_advantage_learning_update = v_st_plus_1_target[i] - q_st_plus_1_target[i, selected_actions[i]]
            # Persistent Advantage Learning or Regular Advantage Learning
            if self.persistent:
                TD_targets[i, batch.actions()[i]] -= self.alpha * min(advantage_learning_update, next_advantage_learning_update)
            else:
                TD_targets[i, batch.actions()[i]] -= self.alpha * advantage_learning_update

            # mixing monte carlo updates
            monte_carlo_target = total_returns[i]
            TD_targets[i, batch.actions()[i]] = (1 - self.monte_carlo_mixing_rate) * TD_targets[i, batch.actions()[i]] \
                                        + self.monte_carlo_mixing_rate * monte_carlo_target

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
