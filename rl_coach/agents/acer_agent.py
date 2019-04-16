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
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace
from rl_coach.utils import eps, last_sample


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
        is weighted using the beta value defined by beta_entropy.
    """
    def __init__(self):
        super().__init__()
        self.apply_gradients_every_x_episodes = 5
        self.num_steps_between_gradient_updates = 5000
        self.ratio_of_replay = 4
        self.num_transitions_to_start_replay = 10000
        self.rate_for_copying_weights_to_target = 0.01
        self.importance_weight_truncation = 10.0
        self.use_trust_region_optimization = True
        self.max_KL_divergence = 1.0
        self.beta_entropy = 0


class ACERNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [QHeadParameters(loss_weight=0.5), ACERPolicyHeadParameters(loss_weight=1.0)]
        self.optimizer_type = 'Adam'
        self.async_training = True
        self.clip_gradients = 40.0
        self.create_target_network = True


class ACERAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=ACERAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters()},
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
        self.V_Values = self.register_signal('Values')
        self.kl_divergence = self.register_signal('KL Divergence')

    def _learn_from_batch(self, batch):

        fetches = [self.networks['main'].online_network.output_heads[1].probability_loss,
                   self.networks['main'].online_network.output_heads[1].bias_correction_loss,
                   self.networks['main'].online_network.output_heads[1].kl_divergence]

        # batch contains a list of transitions to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # get the values for the current states
        Q_values, policy_prob = self.networks['main'].online_network.predict(batch.states(network_keys))
        avg_policy_prob = self.networks['main'].target_network.predict(batch.states(network_keys))[1]
        current_state_values = np.sum(policy_prob * Q_values, axis=1)

        actions = batch.actions()
        num_transitions = batch.size
        Q_head_targets = Q_values

        Q_i = Q_values[np.arange(num_transitions), actions]

        mu = batch.info('all_action_probabilities')
        rho = policy_prob / (mu + eps)
        rho_i = rho[np.arange(batch.size), actions]

        rho_bar = np.minimum(1.0, rho_i)

        if batch.game_overs()[-1]:
            Qret = 0
        else:
            result = self.networks['main'].online_network.predict(last_sample(batch.next_states(network_keys)))
            Qret = np.sum(result[0] * result[1], axis=1)[0]

        for i in reversed(range(num_transitions)):
            Qret = batch.rewards()[i] + self.ap.algorithm.discount * Qret
            Q_head_targets[i, actions[i]] = Qret
            Qret = rho_bar[i] * (Qret - Q_i[i]) + current_state_values[i]

        Q_retrace = Q_head_targets[np.arange(num_transitions), actions]

        # train
        result = self.networks['main'].train_and_sync_networks({**batch.states(network_keys),
                                                                'output_1_0': actions,
                                                                'output_1_1': rho,
                                                                'output_1_2': rho_i,
                                                                'output_1_3': Q_values,
                                                                'output_1_4': Q_retrace,
                                                                'output_1_5': avg_policy_prob},
                                                               [Q_head_targets, current_state_values],
                                                               additional_fetches=fetches)

        for network in self.networks.values():
            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

        # logging
        total_loss, losses, unclipped_grads, fetch_result = result[:4]
        self.q_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])
        self.probability_loss.add_sample(fetch_result[0])
        self.bias_correction_loss.add_sample(fetch_result[1])
        self.unclipped_grads.add_sample(unclipped_grads)
        self.V_Values.add_sample(current_state_values)
        self.kl_divergence.add_sample(fetch_result[2])

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
