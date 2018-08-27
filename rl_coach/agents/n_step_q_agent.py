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

from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.architectures.tensorflow_components.heads.q_head import QHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, AgentParameters, NetworkParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters

from rl_coach.core_types import EnvironmentSteps
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.episodic.single_episode_buffer import SingleEpisodeBufferParameters
from rl_coach.utils import last_sample


class NStepQNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [QHeadParameters()]
        self.loss_weights = [1.0]
        self.optimizer_type = 'Adam'
        self.async_training = True
        self.shared_optimizer = True
        self.create_target_network = True


class NStepQAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(10000)
        self.apply_gradients_every_x_episodes = 1
        self.num_steps_between_gradient_updates = 5  # this is called t_max in all the papers
        self.targets_horizon = 'N-Step'


class NStepQAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=NStepQAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=SingleEpisodeBufferParameters(),
                         networks={"main": NStepQNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.n_step_q_agent:NStepQAgent'


# N Step Q Learning Agent - https://arxiv.org/abs/1602.01783
class NStepQAgent(ValueOptimizationAgent, PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.last_gradient_update_step_idx = 0
        self.q_values = self.register_signal('Q Values')
        self.value_loss = self.register_signal('Value Loss')

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # get the values for the current states
        state_value_head_targets = self.networks['main'].online_network.predict(batch.states(network_keys))

        # the targets for the state value estimator
        if self.ap.algorithm.targets_horizon == '1-Step':
            # 1-Step Q learning
            q_st_plus_1 = self.networks['main'].target_network.predict(batch.next_states(network_keys))

            for i in reversed(range(batch.size)):
                state_value_head_targets[i][batch.actions()[i]] = \
                    batch.rewards()[i] \
                    + (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * np.max(q_st_plus_1[i], 0)

        elif self.ap.algorithm.targets_horizon == 'N-Step':
            # N-Step Q learning
            if batch.game_overs()[-1]:
                R = 0
            else:
                R = np.max(self.networks['main'].target_network.predict(last_sample(batch.next_states(network_keys))))

            for i in reversed(range(batch.size)):
                R = batch.rewards()[i] + self.ap.algorithm.discount * R
                state_value_head_targets[i][batch.actions()[i]] = R

        else:
            assert True, 'The available values for targets_horizon are: 1-Step, N-Step'

        # train
        result = self.networks['main'].online_network.accumulate_gradients(batch.states(network_keys), [state_value_head_targets])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.value_loss.add_sample(losses[0])

        return total_loss, losses, unclipped_grads

    def train(self):
        # update the target network of every network that has a target network
        if any([network.has_target for network in self.networks.values()]) \
                and self._should_update_online_weights_to_target():
            for network in self.networks.values():
                network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

            self.agent_logger.create_signal_value('Update Target Network', 1)
        else:
            self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

        return PolicyOptimizationAgent.train(self)
