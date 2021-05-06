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

import copy
import math
import time
from collections import OrderedDict
from random import shuffle
from typing import Union

import numpy as np

from rl_coach.agents.actor_critic_agent import ActorCriticAgent
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import PPOHeadParameters, VHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, \
    AgentParameters
from rl_coach.core_types import EnvironmentSteps, Batch, EnvResponse, StateType, Transition
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.logger import screen
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace


class ClippedPPONetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='tanh')}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='tanh')
        self.heads_parameters = [VHeadParameters(), PPOHeadParameters()]
        self.batch_size = 64
        self.optimizer_type = 'Adam'
        self.clip_gradients = None
        self.use_separate_networks_per_head = True
        self.async_training = False
        self.l2_regularization = 0

        # The target network is used in order to freeze the old policy, while making updates to the new one
        # in train_network()
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = True


class ClippedPPOAlgorithmParameters(AlgorithmParameters):
    """
    :param policy_gradient_rescaler: (PolicyGradientRescaler)
        This represents how the critic will be used to update the actor. The critic value function is typically used
        to rescale the gradients calculated by the actor. There are several ways for doing this, such as using the
        advantage of the action, or the generalized advantage estimation (GAE) value.

    :param gae_lambda: (float)
        The :math:`\lambda` value is used within the GAE function in order to weight different bootstrap length
        estimations. Typical values are in the range 0.9-1, and define an exponential decay over the different
        n-step estimations.

    :param clip_likelihood_ratio_using_epsilon: (float)
        If not None, the likelihood ratio between the current and new policy in the PPO loss function will be
        clipped to the range [1-clip_likelihood_ratio_using_epsilon, 1+clip_likelihood_ratio_using_epsilon].
        This is typically used in the Clipped PPO version of PPO, and should be set to None in regular PPO
        implementations.

    :param value_targets_mix_fraction: (float)
        The targets for the value network are an exponential weighted moving average which uses this mix fraction to
        define how much of the new targets will be taken into account when calculating the loss.
        This value should be set to the range (0,1], where 1 means that only the new targets will be taken into account.

    :param estimate_state_value_using_gae: (bool)
        If set to True, the state value will be estimated using the GAE technique.

    :param use_kl_regularization: (bool)
        If set to True, the loss function will be regularized using the KL diveregence between the current and new
        policy, to bound the change of the policy during the network update.

    :param beta_entropy: (float)
        An entropy regulaization term can be added to the loss function in order to control exploration. This term
        is weighted using the :math:`\beta` value defined by beta_entropy.

    :param optimization_epochs: (int)
        For each training phase, the collected dataset will be used for multiple epochs, which are defined by the
        optimization_epochs value.

    :param optimization_epochs: (Schedule)
        Can be used to define a schedule over the clipping of the likelihood ratio.

    """
    def __init__(self):
        super().__init__()
        self.num_episodes_in_experience_replay = 1000000
        self.policy_gradient_rescaler = PolicyGradientRescaler.GAE
        self.gae_lambda = 0.95
        self.use_kl_regularization = False
        self.clip_likelihood_ratio_using_epsilon = 0.2
        self.estimate_state_value_using_gae = True
        self.beta_entropy = 0.01  # should be 0 for mujoco
        self.num_consecutive_playing_steps = EnvironmentSteps(2048)
        self.optimization_epochs = 10
        self.clipping_decay_schedule = ConstantSchedule(1)
        self.act_for_full_episodes = True
        self.update_pre_network_filters_state_on_train = True
        self.update_pre_network_filters_state_on_inference = False


class ClippedPPOAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=ClippedPPOAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: AdditiveNoiseParameters()},
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": ClippedPPONetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.clipped_ppo_agent:ClippedPPOAgent'


# Clipped Proximal Policy Optimization - https://arxiv.org/abs/1707.06347
class ClippedPPOAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        # signals definition
        self.value_loss = self.register_signal('Value Loss')
        self.policy_loss = self.register_signal('Policy Loss')
        self.total_kl_divergence_during_training_process = 0.0
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.value_targets = self.register_signal('Value Targets')
        self.kl_divergence = self.register_signal('KL Divergence')
        self.likelihood_ratio = self.register_signal('Likelihood Ratio')
        self.clipped_likelihood_ratio = self.register_signal('Clipped Likelihood Ratio')
        self.policy_std = self.register_signal('Policy Standard Deviation')
        self.noise = np.random.uniform(low=-0.25,
                                       high=0.25)
        self.last_policy_id_networks_synced = 0

    @property
    def is_on_policy(self) -> bool:
        return True

    def initialize_session_dependent_components(self):
        super().initialize_session_dependent_components()

        # TODO get hyper-params from self.ap, and write more nicely
        self.is_LSTM_middleware = self.networks["main"].online_network.middleware[
                                      -1].__class__.__name__ == 'LSTMMiddleware'
        if self.is_LSTM_middleware:
            self.sequence_length = self.networks["main"].online_network.middleware[-1].sequence_length
            self.horizon = self.networks["main"].online_network.middleware[-1].horizon
            if self.horizon < 1:
                raise ValueError("LSTM sequence horizon has to be >= 1")
            self.stride = self.networks["main"].online_network.middleware[-1].stride

    def build_dataset(self):
        update_internal_state = self.ap.algorithm.update_pre_network_filters_state_on_train
        if self.is_LSTM_middleware:
            dataset = []
            for episode in self.memory.get_all_complete_episodes():
                episode.transitions = self.pre_network_filter.filter(episode.transitions,
                                                                     deep_copy=False,
                                                                     update_internal_state=update_internal_state)
                for i in range(0, episode.length()-self.sequence_length, self.stride):
                    trajectory = episode.transitions[i:i + self.sequence_length]
                    transition = Transition()
                    transition.add_info({'sample_sequence': Batch(trajectory)})
                    dataset.append(transition)
            batch_size = self.ap.network_wrappers['main'].batch_size
            dataset = dataset[:batch_size*(len(dataset) // batch_size)]
        else:
            dataset = self.memory.transitions
            dataset = self.pre_network_filter.filter(dataset, deep_copy=False,
                                                     update_internal_state=update_internal_state)
        return Batch(dataset)

    def build_network_inputs(self, data, network_keys, start=None, end=None, with_next_state=True):
        feed_dict = {}
        for k in network_keys:
            if self.is_LSTM_middleware:
                if with_next_state:
                    seq = [np.vstack([t.info['sample_sequence'].states([k])[k],
                                      np.expand_dims(t.info['sample_sequence'].next_states([k])[k][-1], 0)])
                           for t in data[start:end]]
                else:
                    # TODO this part of the flow should be rewritten
                    seq = [np.vstack([t.info['sample_sequence'].states([k])[k][: self.sequence_length - self.horizon + 1]]) for t in data[start:end]]
                feed_dict[k] = np.vstack(seq)
            else:
                feed_dict[k] = data.states([k])[k][start:end]
        return feed_dict

    def fill_advantages(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        state_values = []
        for i in range(int(batch.size / self.ap.network_wrappers['main'].batch_size) + 1):
            self.networks['main'].online_network.reset_internal_memory()
            start = i * self.ap.network_wrappers['main'].batch_size
            end = (i + 1) * self.ap.network_wrappers['main'].batch_size
            if start == batch.size:
                break

            network_input = self.build_network_inputs(batch, network_keys, start, end)
            state_values.append(self.networks['main'].online_network.predict(network_input)[0])

        current_state_values = np.concatenate(state_values)
        self.state_values.add_sample(current_state_values)

        # calculate advantages
        advantages = []
        value_targets = []

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            total_returns = batch.n_step_discounted_rewards()
            advantages = total_returns - current_state_values
        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            if self.is_LSTM_middleware:
                # get bootstraps
                advantages = np.array([])
                value_targets = np.array([])
                for idx in range(batch.size):
                    rollout_state_values = current_state_values[idx * (self.sequence_length + 1):
                                                                (idx + 1) * (self.sequence_length + 1)].squeeze()
                    rollout_advantages, gae_based_value_targets = \
                        self.get_general_advantage_estimation_values(batch[idx].info['sample_sequence'].rewards(),
                                                                     rollout_state_values)
                    advantages = np.append(advantages, rollout_advantages[: self.sequence_length - self.horizon + 1])
                    value_targets = np.append(value_targets,
                                              gae_based_value_targets[: self.sequence_length - self.horizon + 1])
            else:
                # get bootstraps
                episode_start_idx = 0
                advantages = np.array([])
                value_targets = np.array([])
                for idx, game_over in enumerate(batch.game_overs()):
                    if game_over:
                        # get advantages for the rollout
                        value_bootstrapping = np.zeros((1,))
                        rollout_state_values = np.append(current_state_values[episode_start_idx:idx + 1],
                                                         value_bootstrapping)

                        rollout_advantages, gae_based_value_targets = \
                            self.get_general_advantage_estimation_values(batch.rewards()[episode_start_idx:idx + 1],
                                                                         rollout_state_values)
                        episode_start_idx = idx + 1
                        advantages = np.append(advantages, rollout_advantages)
                        value_targets = np.append(value_targets, gae_based_value_targets)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # standardize
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        if self.is_LSTM_middleware:
            for idx, transition in enumerate(batch.transitions):
                transition.info['advantage'] = np.expand_dims(
                    advantages[idx * (self.sequence_length - self.horizon + 1):
                               (idx + 1) * (self.sequence_length - self.horizon + 1)], -1)
                transition.info['gae_based_value_target'] = np.expand_dims(
                    value_targets[idx * (self.sequence_length - self.horizon + 1):
                                  (idx + 1) * (self.sequence_length - self.horizon + 1)], -1)
        else:
            for transition, advantage, value_target in zip(batch.transitions, advantages, value_targets):
                transition.info['advantage'] = advantage
                transition.info['gae_based_value_target'] = value_target

        self.action_advantages.add_sample(advantages)

    def train_network(self, batch, epochs):
        batch_results = []
        for j in range(epochs):
            batch.shuffle()
            batch_results = {
                'total_loss': [],
                'losses': [],
                'unclipped_grads': [],
                'kl_divergence': [],
                'entropy': []
            }

            fetches = [self.networks['main'].online_network.output_heads[1].kl_divergence,
                       self.networks['main'].online_network.output_heads[1].entropy,
                       self.networks['main'].online_network.output_heads[1].likelihood_ratio,
                       self.networks['main'].online_network.output_heads[1].clipped_likelihood_ratio]
            if hasattr(self.networks['main'].online_network.output_heads[1], 'policy_std'):
                fetches.append(self.networks['main'].online_network.output_heads[1].policy_std)

            for i in range(math.ceil(batch.size / self.ap.network_wrappers['main'].batch_size)):
                self.networks['main'].online_network.reset_internal_memory()
                self.networks['main'].target_network.reset_internal_memory()

                start = i * self.ap.network_wrappers['main'].batch_size
                end = (i + 1) * self.ap.network_wrappers['main'].batch_size

                network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()
                if self.is_LSTM_middleware:
                    # TODO move to __init__
                    is_discrete = isinstance(self.spaces.action, DiscreteActionSpace)
                    if is_discrete:
                        actions = np.vstack(
                            [seq.actions(expand_dims=True)[: self.sequence_length - self.horizon + 1] for seq in
                             batch.info('sample_sequence')[start:end]]).squeeze()
                    else:
                        actions = np.vstack([seq.actions()[: self.sequence_length - self.horizon + 1] for seq in
                                             batch.info('sample_sequence')[start:end]])
                    gae_based_value_targets = np.vstack(
                        batch.info_as_list('gae_based_value_target')[start:end]).squeeze()
                    advantages = np.vstack(batch.info_as_list('advantage')[start:end]).squeeze()
                else:
                    actions = batch.actions()[start:end]
                    gae_based_value_targets = batch.info('gae_based_value_target')[start:end]
                    advantages = batch.info('advantage')[start:end]
                if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) == 1:
                    actions = np.expand_dims(actions, -1)

                # get old policy probabilities and distribution

                # TODO-perf - the target network ("old_policy") is not changing. this can be calculated once for all epochs.
                # the shuffling being done, should only be performed on the indices.
                network_input = self.build_network_inputs(batch, network_keys, start, end, with_next_state=False)
                result = self.networks['main'].target_network.predict(network_input)
                old_policy_distribution = result[1:]

                # calculate gradients and apply on both the local policy network and on the global policy network
                if self.ap.algorithm.estimate_state_value_using_gae:
                    value_targets = np.expand_dims(gae_based_value_targets, -1)
                else:
                    total_returns = batch.n_step_discounted_rewards(expand_dims=True)
                    value_targets = total_returns[start:end]

                inputs = copy.copy(network_input)
                inputs['output_1_0'] = actions

                # The old_policy_distribution needs to be represented as a list, because in the event of
                # discrete controls, it has just a mean. otherwise, it has both a mean and standard deviation
                for input_index, input in enumerate(old_policy_distribution):
                    inputs['output_1_{}'.format(input_index + 1)] = input

                # update the clipping decay schedule value
                inputs['output_1_{}'.format(len(old_policy_distribution)+1)] = \
                    self.ap.algorithm.clipping_decay_schedule.current_value

                total_loss, losses, unclipped_grads, fetch_result = \
                    self.networks['main'].train_and_sync_networks(
                        inputs, [value_targets, advantages], additional_fetches=fetches
                    )
                batch_results['total_loss'].append(total_loss)
                batch_results['losses'].append(losses)
                batch_results['unclipped_grads'].append(unclipped_grads)
                batch_results['kl_divergence'].append(fetch_result[0])
                batch_results['entropy'].append(fetch_result[1])

                self.unclipped_grads.add_sample(unclipped_grads)
                self.value_targets.add_sample(value_targets)
                self.likelihood_ratio.add_sample(fetch_result[2])
                self.clipped_likelihood_ratio.add_sample(fetch_result[3])

            last_policy_std = fetch_result[-1][0]

            for key in batch_results.keys():
                batch_results[key] = np.mean(batch_results[key], 0)

            self.value_loss.add_sample(batch_results['losses'][0])
            self.policy_loss.add_sample(batch_results['losses'][1])
            self.loss.add_sample(batch_results['total_loss'])

            if self.ap.network_wrappers['main'].learning_rate_decay_rate != 0:
                curr_learning_rate = self.networks['main'].online_network.get_variable_value(
                    self.networks['main'].online_network.adaptive_learning_rate_scheme)
                self.curr_learning_rate.add_sample(curr_learning_rate)
            else:
                curr_learning_rate = self.ap.network_wrappers['main'].learning_rate

            # log training parameters
            screen.log_dict(
                OrderedDict([
                    ("Surrogate loss", batch_results['losses'][1]),
                    ("KL divergence", batch_results['kl_divergence']),
                    ("Entropy", batch_results['entropy']),
                    ("training epoch", j),
                    ("learning_rate", curr_learning_rate)
                ]),
                prefix="Policy training"
            )
        self.total_kl_divergence_during_training_process = batch_results['kl_divergence']
        self.entropy.add_sample(batch_results['entropy'])
        self.kl_divergence.add_sample(batch_results['kl_divergence'])
        self.policy_std.add_sample(last_policy_std)
        return batch_results['losses']

    def post_training_commands(self):
        # clean memory
        self.call_memory('clean')

    def train(self):
        s = time.time()
        if self._should_train():
            screen.print("*** DEBUG: trainer starts a training iteration ***")
            for network in self.networks.values():
                network.set_is_training(True)

            batch = self.build_dataset()

            new_policy_was_published = self.policy_id > self.last_policy_id_networks_synced
            if not self.ap.is_mast_training or new_policy_was_published:
                # in MAST, the graph manager controls synchronization directly so that the old policy will update
                # only after the policy publish to actors
                self.networks['main'].sync()
                self.last_policy_id_networks_synced = self.policy_id

            self.fill_advantages(batch)

            self.train_network(batch, self.ap.algorithm.optimization_epochs)

            for network in self.networks.values():
                network.set_is_training(False)

            self.post_training_commands()
            self.training_iteration += 1
            # should be done in order to update the data that has been accumulated * while not playing *
            self.update_log()
            screen.print("*** DEBUG: training iter time = {} ***".format(time.time()-s))
            return None

    def run_pre_network_filter_for_inference(self, state: StateType, update_internal_state: bool=False):
        dummy_env_response = EnvResponse(next_state=state, reward=0, game_over=False)
        update_internal_state = self.ap.algorithm.update_pre_network_filters_state_on_inference
        return self.pre_network_filter.filter(
            dummy_env_response, update_internal_state=update_internal_state)[0].next_state

    def choose_action(self, curr_state):
        self.ap.algorithm.clipping_decay_schedule.step()
        return super().choose_action(curr_state)
