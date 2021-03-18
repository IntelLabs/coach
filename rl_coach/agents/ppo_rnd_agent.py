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

from collections import OrderedDict
from random import shuffle
from typing import Union
import numpy as np

from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import PPOHeadParameters, VHeadParameters, RNDHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AgentParameters, MiddlewareScheme
from rl_coach.core_types import Batch, Transition
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.logger import screen
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAlgorithmParameters, ClippedPPOAgent, \
    ClippedPPONetworkParameters
from rl_coach.utilities.shared_running_stats import NumpySharedRunningStats


class PPORNDAlgorithmParameters(ClippedPPOAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.rnd_sample_ratio = 1.0


class PPORNDNetworkParameters(ClippedPPONetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.heads_parameters = [VHeadParameters(), VHeadParameters(), PPOHeadParameters()]


class RNDNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='leaky_relu',
                                                                                  input_rescaling={'image': 1.0})}
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Empty)
        self.heads_parameters = [RNDHeadParameters()]
        self.create_target_network = False
        self.optimizer_type = 'Adam'
        self.batch_size = 1024
        self.learning_rate = 0.0001
        self.should_get_softmax_probabilities = False


class PPORNDAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=PPORNDAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: AdditiveNoiseParameters()},
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": PPORNDNetworkParameters(),
                                   "predictor": RNDNetworkParameters(),
                                   "constant": RNDNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.ppo_rnd_agent:PPORNDAgent'


# Random Network Distillation - https://arxiv.org/pdf/1810.12894.pdf
class PPORNDAgent(ClippedPPOAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.rnd_stats = NumpySharedRunningStats(name='RND_normalization', epsilon=1e-8)
        self.rnd_stats.set_params(clip_values=[-5, 5])
        self.rnd_obs_stats = NumpySharedRunningStats(name='RND_observation_normalization', epsilon=1e-8)

    def prepare_rnd_inputs(self, batch):
        # print(batch.transitions)
        states = batch.next_states(['observation'])['observation'][:, :, :, 0]
        return {'observation': np.expand_dims(self.rnd_obs_stats.normalize(states), -1)}

    def handle_intrinsic_reward_based_batch(self):
        transitions = self.memory.transitions
        batch_size = self.ap.network_wrappers['main'].batch_size
        for i in range(int(self.memory.num_transitions() / batch_size) + 1):
            start = i * batch_size
            end = (i + 1) * batch_size
            if start == self.memory.num_transitions():
                break

            inputs = self.prepare_rnd_inputs(Batch(transitions[start:end]))
            embedding = self.networks['constant'].online_network.predict(inputs)
            prediction = self.networks['predictor'].online_network.predict(inputs)
            prediction_error = np.sum((embedding - prediction) ** 2, axis=1)
            self.rnd_stats.push_val(np.expand_dims(prediction_error, -1))
            intrinsic_rewards = self.rnd_stats.normalize(prediction_error)

            for i, transition in enumerate(transitions[start:end]):
                transition.reward = [np.sign(transition.reward), intrinsic_rewards[i]]
                transition.game_over = [transition.game_over, False]

    def update_transition_before_adding_to_replay_buffer(self, transition: Transition):
        observation = np.array(transition.state['observation'])
        if self.rnd_obs_stats.n < 1:
            self.rnd_obs_stats.set_params(shape=observation[:, :, 0].shape, clip_values=[-5, 5])
        self.rnd_obs_stats.push_val(np.expand_dims(observation[:, :, 0], 0))
        return transition

    def train_rnd(self):
        transitions = list(np.random.choice(self.memory.transitions,
                                            int(self.memory.num_transitions() * self.ap.algorithm.rnd_sample_ratio),
                                            replace=False))
        dataset = Batch(transitions)
        dataset_order = list(range(dataset.size))
        batch_size = self.ap.network_wrappers['main'].batch_size
        for epoch in range(self.ap.algorithm.optimization_epochs):
            shuffle(dataset_order)
            total_loss = 0
            total_grads = 0
            for i in range(int(dataset.size / batch_size)):
                start = i * batch_size
                end = (i + 1) * batch_size

                batch = Batch(list(np.array(dataset.transitions)[dataset_order[start:end]]))
                inputs = self.prepare_rnd_inputs(batch)

                const_embedding = self.networks['constant'].online_network.predict(inputs)

                res = self.networks['predictor'].train_and_sync_networks(inputs, [const_embedding])

                total_loss += res[0]
                total_grads += res[2]

            screen.log_dict(
                OrderedDict([
                    ("training epoch", epoch),
                    ("dataset size", dataset.size),
                    ("mean loss", total_loss / dataset.size),
                    ("mean gradients", total_grads / dataset.size)
                ]),
                prefix="RND Training"
            )

    def pre_training_commands(self):
        self.handle_intrinsic_reward_based_batch()
        self.train_rnd()

