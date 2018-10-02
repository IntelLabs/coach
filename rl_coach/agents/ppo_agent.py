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
from collections import OrderedDict
from typing import Union

import numpy as np

from rl_coach.agents.actor_critic_agent import ActorCriticAgent
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.architectures.tensorflow_components.heads.ppo_head import PPOHeadParameters
from rl_coach.architectures.tensorflow_components.heads.v_head import VHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, \
    AgentParameters, DistributedTaskParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters

from rl_coach.core_types import EnvironmentSteps, Batch
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.logger import screen
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.utils import force_list


class PPOCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='tanh')}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='tanh')
        self.heads_parameters = [VHeadParameters()]
        self.async_training = True
        self.l2_regularization = 0
        self.create_target_network = True
        self.batch_size = 128


class PPOActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='tanh')}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='tanh')
        self.heads_parameters = [PPOHeadParameters()]
        self.optimizer_type = 'Adam'
        self.async_training = True
        self.l2_regularization = 0
        self.create_target_network = True
        self.batch_size = 128


class PPOAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.policy_gradient_rescaler = PolicyGradientRescaler.GAE
        self.gae_lambda = 0.96
        self.target_kl_divergence = 0.01
        self.initial_kl_coefficient = 1.0
        self.high_kl_penalty_coefficient = 1000
        self.clip_likelihood_ratio_using_epsilon = None
        self.value_targets_mix_fraction = 0.1
        self.estimate_state_value_using_gae = True
        self.step_until_collecting_full_episodes = True
        self.use_kl_regularization = True
        self.beta_entropy = 0.01
        self.num_consecutive_playing_steps = EnvironmentSteps(5000)


class PPOAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=PPOAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: AdditiveNoiseParameters()},
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"critic": PPOCriticNetworkParameters(), "actor": PPOActorNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.ppo_agent:PPOAgent'


# Proximal Policy Optimization - https://arxiv.org/pdf/1707.06347.pdf
class PPOAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

        # signals definition
        self.value_loss = self.register_signal('Value Loss')
        self.policy_loss = self.register_signal('Policy Loss')
        self.kl_divergence = self.register_signal('KL Divergence')
        self.total_kl_divergence_during_training_process = 0.0
        self.unclipped_grads = self.register_signal('Grads (unclipped)')

    def fill_advantages(self, batch):
        batch = Batch(batch)
        network_keys = self.ap.network_wrappers['critic'].input_embedders_parameters.keys()

        # * Found not to have any impact *
        # current_states_with_timestep = self.concat_state_and_timestep(batch)

        current_state_values = self.networks['critic'].online_network.predict(batch.states(network_keys)).squeeze()

        # calculate advantages
        advantages = []
        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            advantages = batch.total_returns() - current_state_values
        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            episode_start_idx = 0
            advantages = np.array([])
            # current_state_values[batch.game_overs()] = 0
            for idx, game_over in enumerate(batch.game_overs()):
                if game_over:
                    # get advantages for the rollout
                    value_bootstrapping = np.zeros((1,))
                    rollout_state_values = np.append(current_state_values[episode_start_idx:idx+1], value_bootstrapping)

                    rollout_advantages, _ = \
                        self.get_general_advantage_estimation_values(batch.rewards()[episode_start_idx:idx+1],
                                                                     rollout_state_values)
                    episode_start_idx = idx + 1
                    advantages = np.append(advantages, rollout_advantages)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # standardize
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        # TODO: this will be problematic with a shared memory
        for transition, advantage in zip(self.memory.transitions, advantages):
            transition.info['advantage'] = advantage

        self.action_advantages.add_sample(advantages)

    def train_value_network(self, dataset, epochs):
        loss = []
        batch = Batch(dataset)
        network_keys = self.ap.network_wrappers['critic'].input_embedders_parameters.keys()

        # * Found not to have any impact *
        # add a timestep to the observation
        # current_states_with_timestep = self.concat_state_and_timestep(dataset)

        mix_fraction = self.ap.algorithm.value_targets_mix_fraction
        for j in range(epochs):
            curr_batch_size = batch.size
            if self.networks['critic'].online_network.optimizer_type != 'LBFGS':
                curr_batch_size = self.ap.network_wrappers['critic'].batch_size
            for i in range(batch.size // curr_batch_size):
                # split to batches for first order optimization techniques
                current_states_batch = {
                    k: v[i * curr_batch_size:(i + 1) * curr_batch_size]
                    for k, v in batch.states(network_keys).items()
                }
                total_return_batch = batch.total_returns(True)[i * curr_batch_size:(i + 1) * curr_batch_size]
                old_policy_values = force_list(self.networks['critic'].target_network.predict(
                    current_states_batch).squeeze())
                if self.networks['critic'].online_network.optimizer_type != 'LBFGS':
                    targets = total_return_batch
                else:
                    current_values = self.networks['critic'].online_network.predict(current_states_batch)
                    targets = current_values * (1 - mix_fraction) + total_return_batch * mix_fraction

                inputs = copy.copy(current_states_batch)
                for input_index, input in enumerate(old_policy_values):
                    name = 'output_0_{}'.format(input_index)
                    if name in self.networks['critic'].online_network.inputs:
                        inputs[name] = input

                value_loss = self.networks['critic'].online_network.accumulate_gradients(inputs, targets)

                self.networks['critic'].apply_gradients_to_online_network()
                if isinstance(self.ap.task_parameters, DistributedTaskParameters):
                    self.networks['critic'].apply_gradients_to_global_network()
                self.networks['critic'].online_network.reset_accumulated_gradients()

                loss.append([value_loss[0]])
        loss = np.mean(loss, 0)
        return loss

    def concat_state_and_timestep(self, dataset):
        current_states_with_timestep = [np.append(transition.state['observation'], transition.info['timestep'])
                                        for transition in dataset]
        current_states_with_timestep = np.expand_dims(current_states_with_timestep, -1)
        return current_states_with_timestep

    def train_policy_network(self, dataset, epochs):
        loss = []
        for j in range(epochs):
            loss = {
                'total_loss': [],
                'policy_losses': [],
                'unclipped_grads': [],
                'fetch_result': []
            }
            #shuffle(dataset)
            for i in range(len(dataset) // self.ap.network_wrappers['actor'].batch_size):
                batch = Batch(dataset[i * self.ap.network_wrappers['actor'].batch_size:
                                      (i + 1) * self.ap.network_wrappers['actor'].batch_size])

                network_keys = self.ap.network_wrappers['actor'].input_embedders_parameters.keys()

                advantages = batch.info('advantage')
                actions = batch.actions()
                if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) == 1:
                    actions = np.expand_dims(actions, -1)

                # get old policy probabilities and distribution
                old_policy = force_list(self.networks['actor'].target_network.predict(batch.states(network_keys)))

                # calculate gradients and apply on both the local policy network and on the global policy network
                fetches = [self.networks['actor'].online_network.output_heads[0].kl_divergence,
                           self.networks['actor'].online_network.output_heads[0].entropy]

                inputs = copy.copy(batch.states(network_keys))
                inputs['output_0_0'] = actions

                # old_policy_distribution needs to be represented as a list, because in the event of discrete controls,
                # it has just a mean. otherwise, it has both a mean and standard deviation
                for input_index, input in enumerate(old_policy):
                    inputs['output_0_{}'.format(input_index + 1)] = input

                total_loss, policy_losses, unclipped_grads, fetch_result =\
                    self.networks['actor'].online_network.accumulate_gradients(
                        inputs, [advantages], additional_fetches=fetches)

                self.networks['actor'].apply_gradients_to_online_network()
                if isinstance(self.ap.task_parameters, DistributedTaskParameters):
                    self.networks['actor'].apply_gradients_to_global_network()

                self.networks['actor'].online_network.reset_accumulated_gradients()

                loss['total_loss'].append(total_loss)
                loss['policy_losses'].append(policy_losses)
                loss['unclipped_grads'].append(unclipped_grads)
                loss['fetch_result'].append(fetch_result)

                self.unclipped_grads.add_sample(unclipped_grads)

            for key in loss.keys():
                loss[key] = np.mean(loss[key], 0)

            if self.ap.network_wrappers['critic'].learning_rate_decay_rate != 0:
                curr_learning_rate = self.networks['critic'].online_network.get_variable_value(self.ap.learning_rate)
                self.curr_learning_rate.add_sample(curr_learning_rate)
            else:
                curr_learning_rate = self.ap.network_wrappers['critic'].learning_rate

            # log training parameters
            screen.log_dict(
                OrderedDict([
                    ("Surrogate loss", loss['policy_losses'][0]),
                    ("KL divergence", loss['fetch_result'][0]),
                    ("Entropy", loss['fetch_result'][1]),
                    ("training epoch", j),
                    ("learning_rate", curr_learning_rate)
                ]),
                prefix="Policy training"
            )

        self.total_kl_divergence_during_training_process = loss['fetch_result'][0]
        self.entropy.add_sample(loss['fetch_result'][1])
        self.kl_divergence.add_sample(loss['fetch_result'][0])
        return loss['total_loss']

    def update_kl_coefficient(self):
        # John Schulman takes the mean kl divergence only over the last epoch which is strange but we will follow
        # his implementation for now because we know it works well
        screen.log_title("KL = {}".format(self.total_kl_divergence_during_training_process))

        # update kl coefficient
        kl_target = self.ap.algorithm.target_kl_divergence
        kl_coefficient = self.networks['actor'].online_network.get_variable_value(
            self.networks['actor'].online_network.output_heads[0].kl_coefficient)
        new_kl_coefficient = kl_coefficient
        if self.total_kl_divergence_during_training_process > 1.3 * kl_target:
            # kl too high => increase regularization
            new_kl_coefficient *= 1.5
        elif self.total_kl_divergence_during_training_process < 0.7 * kl_target:
            # kl too low => decrease regularization
            new_kl_coefficient /= 1.5

        # update the kl coefficient variable
        if kl_coefficient != new_kl_coefficient:
            self.networks['actor'].online_network.set_variable_value(
                self.networks['actor'].online_network.output_heads[0].assign_kl_coefficient,
                new_kl_coefficient,
                self.networks['actor'].online_network.output_heads[0].kl_coefficient_ph)

        screen.log_title("KL penalty coefficient change = {} -> {}".format(kl_coefficient, new_kl_coefficient))

    def post_training_commands(self):
        if self.ap.algorithm.use_kl_regularization:
            self.update_kl_coefficient()

        # clean memory
        self.call_memory('clean')

    def train(self):
        loss = 0
        if self._should_train(wait_for_full_episode=True):
            for network in self.networks.values():
                network.set_is_training(True)

            for training_step in range(self.ap.algorithm.num_consecutive_training_steps):
                self.networks['actor'].sync()
                self.networks['critic'].sync()

                dataset = self.memory.transitions

                self.fill_advantages(dataset)

                # take only the requested number of steps
                dataset = dataset[:self.ap.algorithm.num_consecutive_playing_steps.num_steps]

                value_loss = self.train_value_network(dataset, 1)
                policy_loss = self.train_policy_network(dataset, 10)

                self.value_loss.add_sample(value_loss)
                self.policy_loss.add_sample(policy_loss)

            for network in self.networks.values():
                network.set_is_training(False)

            self.post_training_commands()
            self.training_iteration += 1
            self.update_log()  # should be done in order to update the data that has been accumulated * while not playing *
            return np.append(value_loss, policy_loss)

    def get_prediction(self, states):
        tf_input_state = self.prepare_batch_for_inference(states, "actor")
        return self.networks['actor'].online_network.predict(tf_input_state)
