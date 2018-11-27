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

from rl_coach.agents.categorical_dqn_agent import CategoricalDQNAlgorithmParameters, \
    CategoricalDQNAgent, CategoricalDQNAgentParameters
from rl_coach.agents.dqn_agent import DQNNetworkParameters
from rl_coach.architectures.head_parameters import RainbowQHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.exploration_policies.parameter_noise import ParameterNoiseParameters
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters, \
    PrioritizedExperienceReplay


class RainbowDQNNetworkParameters(DQNNetworkParameters):
    def __init__(self):
        super().__init__()
        self.heads_parameters = [RainbowQHeadParameters()]
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Empty)


class RainbowDQNAlgorithmParameters(CategoricalDQNAlgorithmParameters):
    """
    :param n_step: (int)
        The number of steps to bootstrap the network over. The first N-1 steps actual rewards will be accumulated
        using an exponentially growing discount factor, and the Nth step will be bootstrapped from the network
        prediction.

    :param store_transitions_only_when_episodes_are_terminated: (bool)
        If set to True, the transitions will be stored in an Episode object until the episode ends, and just then
        written to the memory. This is useful since we want to calculate the N-step discounted rewards before saving the
        transitions into the memory, and to do so we need the entire episode first.
    """
    def __init__(self):
        super().__init__()
        self.n_step = 3

        # needed for n-step updates to work. i.e. waiting for a full episode to be closed before storing each transition
        self.store_transitions_only_when_episodes_are_terminated = True


class RainbowDQNAgentParameters(CategoricalDQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = RainbowDQNAlgorithmParameters()
        self.exploration = ParameterNoiseParameters(self)
        self.memory = PrioritizedExperienceReplayParameters()
        self.network_wrappers = {"main": RainbowDQNNetworkParameters()}

    @property
    def path(self):
        return 'rl_coach.agents.rainbow_dqn_agent:RainbowDQNAgent'


# Rainbow Deep Q Network - https://arxiv.org/abs/1710.02298
# Agent implementation is composed of:
# 1. NoisyNets
# 2. C51
# 3. Prioritized ER
# 4. DDQN
# 5. Dueling DQN
# 6. N-step returns

class RainbowDQNAgent(CategoricalDQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        ddqn_selected_actions = np.argmax(self.distribution_prediction_to_q_values(
            self.networks['main'].online_network.predict(batch.next_states(network_keys))), axis=1)

        # for the action we actually took, the error is calculated by the atoms distribution
        # for all other actions, the error is 0
        distributional_q_st_plus_n, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        # only update the action that we have actually done in this transition (using the Double-DQN selected actions)
        target_actions = ddqn_selected_actions
        m = np.zeros((self.ap.network_wrappers['main'].batch_size, self.z_values.size))

        batches = np.arange(self.ap.network_wrappers['main'].batch_size)
        for j in range(self.z_values.size):
            # we use batch.info('should_bootstrap_next_state') instead of (1 - batch.game_overs()) since with n-step,
            # we will not bootstrap for the last n-step transitions in the episode
            tzj = np.fmax(np.fmin(batch.n_step_discounted_rewards() + batch.info('should_bootstrap_next_state') *
                                  (self.ap.algorithm.discount ** self.ap.algorithm.n_step) * self.z_values[j],
                                  self.z_values[-1]), self.z_values[0])
            bj = (tzj - self.z_values[0])/(self.z_values[1] - self.z_values[0])
            u = (np.ceil(bj)).astype(int)
            l = (np.floor(bj)).astype(int)
            m[batches, l] += (distributional_q_st_plus_n[batches, target_actions, j] * (u - bj))
            m[batches, u] += (distributional_q_st_plus_n[batches, target_actions, j] * (bj - l))

        # total_loss = cross entropy between actual result above and predicted result for the given action
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

