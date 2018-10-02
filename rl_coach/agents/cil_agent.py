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

from rl_coach.agents.imitation_agent import ImitationAgent
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.heads.cil_head import RegressionHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AgentParameters, MiddlewareScheme, NetworkParameters, AlgorithmParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.non_episodic.balanced_experience_replay import BalancedExperienceReplayParameters


class CILAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.collect_new_data = False
        self.state_key_with_the_class_index = 'high_level_command'


class CILNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Medium)
        self.heads_parameters = [RegressionHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 32
        self.replace_mse_with_huber_loss = False
        self.create_target_network = False


class CILAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=CILAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=BalancedExperienceReplayParameters(),
                         networks={"main": CILNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.cil_agent:CILAgent'


# Conditional Imitation Learning Agent: https://arxiv.org/abs/1710.02410
class CILAgent(ImitationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.current_high_level_control = 0

    def choose_action(self, curr_state):
        self.current_high_level_control = curr_state[self.ap.algorithm.state_key_with_the_class_index]
        return super().choose_action(curr_state)

    def extract_action_values(self, prediction):
        return prediction[self.current_high_level_control].squeeze()

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        target_values = self.networks['main'].online_network.predict({**batch.states(network_keys)})

        branch_to_update = batch.states([self.ap.algorithm.state_key_with_the_class_index])[self.ap.algorithm.state_key_with_the_class_index]
        for idx, branch in enumerate(branch_to_update):
            target_values[branch][idx] = batch.actions()[idx]

        result = self.networks['main'].train_and_sync_networks({**batch.states(network_keys)}, target_values)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
