#
# Copyright (c) 2019 Intel Corporation 
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

import copy
from typing import Union
from collections import OrderedDict
import numpy as np

from rl_coach.agents.ddpg_agent import DDPGAlgorithmParameters, DDPGActorNetworkParameters, \
    DDPGCriticNetworkParameters, DDPGAgent
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionInfo
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.memories.non_episodic.differentiable_neural_dictionary import AnnoyDictionary
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.architectures.head_parameters import WolpertingerActorHeadParameters


class WolpertingerCriticNetworkParameters(DDPGCriticNetworkParameters):
    def __init__(self, use_batchnorm=False):
        super().__init__(use_batchnorm=use_batchnorm)


class WolpertingerActorNetworkParameters(DDPGActorNetworkParameters):
    def __init__(self, use_batchnorm=False):
        super().__init__()
        self.heads_parameters = [WolpertingerActorHeadParameters(batchnorm=use_batchnorm)]


class WolpertingerAlgorithmParameters(DDPGAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.action_embedding_width = 1
        self.k = 1
        

class WolpertingerAgentParameters(AgentParameters):
    def __init__(self, use_batchnorm=False):
        exploration_params = AdditiveNoiseParameters()
        exploration_params.noise_as_percentage_from_action_space = False

        super().__init__(algorithm=WolpertingerAlgorithmParameters(),
                         exploration=exploration_params,
                         memory=EpisodicExperienceReplayParameters(),
                         networks=OrderedDict(
                             [("actor", WolpertingerActorNetworkParameters(use_batchnorm=use_batchnorm)),
                              ("critic", WolpertingerCriticNetworkParameters(use_batchnorm=use_batchnorm))]))

    @property
    def path(self):
        return 'rl_coach.agents.wolpertinger_agent:WolpertingerAgent'


# Deep Reinforcement Learning in Large Discrete Action Spaces - https://arxiv.org/pdf/1512.07679.pdf
class WolpertingerAgent(DDPGAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent'] = None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        # replay buffer holds the actions in the discrete manner, as the agent is expected to act with discrete actions
        # with the BoxDiscretization output filter. But DDPG needs to work on continuous actions, thus converting to
        # continuous actions. This is actually a duplicate since this filtering is also done before applying actions on
        # the environment. So might want to somehow reuse that conversion. Maybe can hold this information in the info
        # dictionary of the transition.

        output_action_filter = \
            list(self.output_filter.action_filters.values())[0]
        continuous_actions = []
        for action in batch.actions():
            continuous_actions.append(output_action_filter.filter(action))
        batch._actions = np.array(continuous_actions).squeeze()

        return super().learn_from_batch(batch)

    def train(self):
        return super().train()

    def choose_action(self, curr_state):
        if not isinstance(self.spaces.action, DiscreteActionSpace):
            raise ValueError("WolpertingerAgent works only for discrete control problems")

        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'actor')
        actor_network = self.networks['actor'].online_network
        critic_network = self.networks['critic'].online_network
        proto_action = actor_network.predict(tf_input_state)
        proto_action = np.expand_dims(self.exploration_policy.get_action(proto_action), 0)

        nn_action_embeddings, indices, _, _ = self.knn_tree.query(keys=proto_action, k=self.ap.algorithm.k)

        # now move the actions through the critic and choose the one with the highest q value
        critic_inputs = copy.copy(tf_input_state)
        critic_inputs['observation'] = np.tile(critic_inputs['observation'], (self.ap.algorithm.k, 1))
        critic_inputs['action'] = nn_action_embeddings[0]
        q_values = critic_network.predict(critic_inputs)[0]
        action = int(indices[0][np.argmax(q_values)])
        self.action_signal.add_sample(action)
        return ActionInfo(action=action, action_value=0)

    def init_environment_dependent_modules(self):
        super().init_environment_dependent_modules()
        self.knn_tree = self.get_initialized_knn()

    # TODO - ideally the knn should not be defined here, but somehow be defined by the user in the preset
    def get_initialized_knn(self):
        num_actions = len(self.spaces.action.actions)
        action_max_abs_range = self.spaces.action.filtered_action_space.max_abs_range if \
            (hasattr(self.spaces.action, 'filtered_action_space') and
             isinstance(self.spaces.action.filtered_action_space, BoxActionSpace)) \
            else 1.0
        keys = np.expand_dims((np.arange(num_actions) / (num_actions - 1) - 0.5) * 2, 1) * action_max_abs_range
        values = np.expand_dims(np.arange(num_actions), 1)
        knn_tree = AnnoyDictionary(dict_size=num_actions, key_width=self.ap.algorithm.action_embedding_width)
        knn_tree.add(keys, values, force_rebuild_tree=True)

        return knn_tree

