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

from rl_coach.agents.imitation_agent import ImitationAgent
from rl_coach.architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AgentParameters, AlgorithmParameters, NetworkParameters, \
    MiddlewareScheme
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters


class BCAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.collect_new_data = False


class BCNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Medium)
        self.heads_parameters = [PolicyHeadParameters()]
        self.loss_weights = [1.0]
        self.optimizer_type = 'Adam'
        self.batch_size = 32
        self.replace_mse_with_huber_loss = False
        self.create_target_network = False


class BCAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=BCAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": BCNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.bc_agent:BCAgent'


# Behavioral Cloning Agent
class BCAgent(ImitationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # When using a policy head, the targets refer to the advantages that we are normally feeding the head with.
        # In this case, we need the policy head to just predict probabilities, so while we usually train the network
        # with log(Pi)*Advantages, in this specific case we will train it to log(Pi), which after the softmax will
        # predict Pi (=probabilities)
        targets = np.ones(batch.actions().shape[0])

        result = self.networks['main'].train_and_sync_networks({**batch.states(network_keys),
                                                                'output_0_0': batch.actions()},
                                                               targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

