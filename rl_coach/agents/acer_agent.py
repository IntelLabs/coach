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
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import ACERPolicyHeadParameters, ACERQHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, AgentParameters
from rl_coach.core_types import Batch
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.utils import get_by_index, eps


class ACERNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [ACERQHeadParameters(loss_weight=0.5), ACERPolicyHeadParameters(loss_weight=1.0)]
        self.async_training = True
        self.clip_gradients = 10.0
        self.create_target_network = True
        self.shared_optimizer = True
        # self.optimizer_type = 'RMSProp'
        # self.optimizer_epsilon = 1e-5
        # self.rms_prop_optimizer_decay = 0.99
        # self.learning_rate = 7e-4
        # self.learning_rate_decay_rate = 0.999
        # self.learning_rate_decay_steps = 1000


class ACERAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_gradient_updates = 20
        self.ratio_of_replay = 4
        self.num_transitions_to_start_replay = 10000
        self.rate_for_copying_weights_to_target = 0.99
        self.importance_weight_truncation = 10.0
        self.use_trust_region_optimization = True
        self.max_KL_divergence = 1
        self.act_for_full_episodes = True
        self.beta_entropy = 0.01


class ACERAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=ACERAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: AdditiveNoiseParameters()},
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": ACERNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.acer_agent:ACERAgent'


# Actor-Critic with Experience Replay - https://arxiv.org/abs/1611.01224
class ACERAgent(PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        # signals definition
        self.q_loss = self.register_signal('Q Loss')
        self.policy_loss = self.register_signal('Policy Loss')
        self.prob_loss = self.register_signal('Probability Loss')
        self.bc_loss = self.register_signal('Bias Correction Loss')
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.Q_targets = self.register_signal('Q Targets')
        self.kl_divergence = self.register_signal('KL Divergence')

    def calc_Q_retrace(self, batch, Q_i, V):
        """
        Calculates Q_retrace targets
        :param batch: a batch of transitions to calculate targets for
        :param Q_values: Q values for the states
        :param V_values: V values for the states
        :return: Q_retrace values
        """
        rho_i = batch.info('rho_i')
        # rho_bar = np.minimum(self.ap.algorithm.importance_weight_truncation, rho_i)
        rho_bar = np.minimum(1.0, rho_i)
        rewards = batch.rewards()
        game_overs = batch.game_overs()
        V_final = V[-1]

        Qret = V_final
        Qrets = []
        nsteps = batch.size
        for i in range(nsteps - 1, -1, -1):
            Qret = rewards[i] + self.ap.algorithm.discount * Qret * (1.0 - game_overs[i])
            Qrets.append(Qret)
            Qret = (rho_bar[i] * (Qret - Q_i[i])) + V[i]
        return np.array(Qrets)

    def _learn_from_batch(self, batch):

        fetches = [self.networks['main'].online_network.output_heads[1].kl_divergence,
                   self.networks['main'].online_network.output_heads[1].prob_loss,
                   self.networks['main'].online_network.output_heads[1].bc_loss]

        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        Q_values, policy_prob = self.networks['main'].online_network.predict(batch.states(network_keys))
        V_values = np.sum(policy_prob * Q_values, axis=1)
        avg_policy_prob = self.networks['main'].target_network.predict(batch.states(network_keys))[1]

        actions = batch.actions()
        Q_i = get_by_index(Q_values, actions)

        mu = batch.info('action_probability')
        rho = policy_prob / (mu + eps)
        for transition, r, action in zip(batch.transitions, rho, actions):
            transition.info['rho'] = r
            transition.info['rho_i'] = r[action]

        Qret = self.calc_Q_retrace(batch, Q_i, V_values)

        advantages = Qret - V_values
        advantages_bc = Q_values - np.expand_dims(V_values, 1)

        total_loss, losses, unclipped_grads, fetch_result = \
            self.networks['main'].train_and_sync_networks(inputs={**batch.states(network_keys),
                                                                       'output_0_0': actions,
                                                                       'output_1_0': actions,
                                                                       'output_1_1': avg_policy_prob},
                                                          targets=[Qret, advantages, advantages_bc],
                                                          importance_weights=[None, batch.info('rho_i'),
                                                                              batch.info('rho')],
                                                          additional_fetches=fetches)

        self.q_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])
        self.prob_loss.add_sample(fetch_result[1])
        self.bc_loss.add_sample(fetch_result[2])
        self.unclipped_grads.add_sample(unclipped_grads)
        self.Q_targets.add_sample(Qret)
        self.kl_divergence.add_sample(fetch_result[0])

        return total_loss, losses, unclipped_grads

    def learn_from_batch(self, batch):
        total_loss, losses, unclipped_grads = self._learn_from_batch(batch)
        if self.ap.algorithm.ratio_of_replay > 0 \
                and self.memory.num_transitions() > self.ap.algorithm.num_transitions_to_start_replay:
            n = np.random.poisson(self.ap.algorithm.ratio_of_replay)
            for _ in range(n):
                new_batch = Batch(self.call_memory('sample', (self.ap.algorithm.num_steps_between_gradient_updates, True)))
                result = self._learn_from_batch(new_batch)
                total_loss += result[0]
                losses += result[1]
                unclipped_grads += result[2]

        return total_loss, losses, unclipped_grads

    def get_prediction(self, states):
        tf_input_state = self.prepare_batch_for_inference(states, "main")
        return self.networks['main'].online_network.predict(tf_input_state)[1:]  # index 0 is the state value
