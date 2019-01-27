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
from rl_coach.architectures.head_parameters import ACERPolicyHeadParameters, QHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, AgentParameters
from rl_coach.core_types import Batch
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.utils import eps


class ACERAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_gradient_updates: (int)
        Every num_steps_between_gradient_updates transitions will be considered as a single batch and use for
        accumulating gradients. This is also the number of steps used for bootstrapping according to the n-step formulation.

    :param ratio_of_replay: (int)
        The number of off-policy training iterations in each ACER iteration.

    :param num_transitions_to_start_replay: (int)
        Number of environment steps until ACER starts to train off-policy from the experience replay.
        This emulates a heat-up phase where the agents learns only on-policy until there are enough transitions in
        the experience replay to start the off-policy training.

    :param rate_for_copying_weights_to_target: (float)
        The rate of the exponential moving average for the average policy which is used for the trust region optimization.
        The target network in this algorithm is used as the average policy.

    :param importance_weight_truncation: (float)
        The clipping constant for the importance weight truncation (not used in the Q-retrace calculation).

    :param use_trust_region_optimization: (bool)
        If set to True, the gradients of the network will be modified with a term dependant on the KL divergence between
        the average policy and the current one, to bound the change of the policy during the network update.

    :param max_KL_divergence: (float)
        The upper bound parameter for the trust region optimization, use_trust_region_optimization needs to be set true
        for this parameter to have an effect.

    :param beta_entropy: (float)
        An entropy regulaization term can be added to the loss function in order to control exploration. This term
        is weighted using the :math:`\beta` value defined by beta_entropy.
    """
    def __init__(self):
        super().__init__()
        self.num_steps_between_gradient_updates = 20
        self.ratio_of_replay = 4
        self.num_transitions_to_start_replay = 10000
        self.rate_for_copying_weights_to_target = 0.99
        self.importance_weight_truncation = 10.0
        self.use_trust_region_optimization = True
        self.max_KL_divergence = 1.0
        self.beta_entropy = 0.01


class ACERNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [QHeadParameters(loss_weight=0.5), ACERPolicyHeadParameters(loss_weight=1.0)]
        self.async_training = True
        self.clip_gradients = 10.0
        self.create_target_network = True
        self.shared_optimizer = True


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
        self.probability_loss = self.register_signal('Probability Loss')
        self.bias_correction_loss = self.register_signal('Bias Correction Loss')
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.Q_targets = self.register_signal('Q Targets')
        self.kl_divergence = self.register_signal('KL Divergence')

    def calc_Q_retrace(self, batch, Q_i, V, rho_i):
        """
        Calculates Q_retrace targets
        :param batch: a batch of transitions to calculate targets for
        :param Q_values: Q values for the states
        :param V_values: V values for the states
        :return: Q_retrace values
        """
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
                   self.networks['main'].online_network.output_heads[1].probability_loss,
                   self.networks['main'].online_network.output_heads[1].bias_correction_loss]

        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        Q_values, policy_prob = self.networks['main'].online_network.predict(batch.states(network_keys))
        V_values = np.sum(policy_prob * Q_values, axis=1)
        avg_policy_prob = self.networks['main'].target_network.predict(batch.states(network_keys))[1]

        actions = batch.actions()
        Q_i = Q_values[np.arange(batch.size), actions]

        mu = batch.info('all_action_probabilities')
        rho = policy_prob / (mu + eps)
        rho_i = rho[np.arange(batch.size), actions]

        Q_retrace = self.calc_Q_retrace(batch, Q_i, V_values, rho_i)
        Q_targets = Q_values
        Q_targets[np.arange(batch.size), actions] = Q_retrace

        total_loss, losses, unclipped_grads, fetch_result = \
            self.networks['main'].train_and_sync_networks(inputs={**batch.states(network_keys),
                                                                  'output_1_0': actions,
                                                                  'output_1_1': rho,
                                                                  'output_1_2': rho_i,
                                                                  'output_1_3': Q_values,
                                                                  'output_1_4': Q_retrace,
                                                                  'output_1_5': avg_policy_prob},
                                                          targets=[Q_targets, V_values],
                                                          additional_fetches=fetches)

        for network in self.networks.values():
            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

        self.q_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])
        self.probability_loss.add_sample(fetch_result[1])
        self.bias_correction_loss.add_sample(fetch_result[2])
        self.unclipped_grads.add_sample(unclipped_grads)
        self.Q_targets.add_sample(Q_retrace)
        self.kl_divergence.add_sample(fetch_result[0])

        return total_loss, losses, unclipped_grads

    def learn_from_batch(self, batch):
        # perform on-policy training iteration
        total_loss, losses, unclipped_grads = self._learn_from_batch(batch)

        if self.ap.algorithm.ratio_of_replay > 0 \
                and self.memory.num_transitions() > self.ap.algorithm.num_transitions_to_start_replay:
            n = np.random.poisson(self.ap.algorithm.ratio_of_replay)
            # perform n off-policy training iterations
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
