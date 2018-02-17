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

from agents.actor_critic_agent import *
from random import shuffle


# Proximal Policy Optimization - https://arxiv.org/pdf/1707.06347.pdf
class PPOAgent(ActorCriticAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ActorCriticAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id,
                                  create_target_network=True)
        self.critic_network = self.main_network

        # define the policy network
        tuning_parameters.agent.input_types = {'observation': InputTypes.Observation}
        tuning_parameters.agent.output_types = [OutputTypes.PPO]
        tuning_parameters.agent.optimizer_type = 'Adam'
        tuning_parameters.agent.l2_regularization = 0
        self.policy_network = NetworkWrapper(tuning_parameters, True, self.has_global, 'policy',
                                             self.replicated_device, self.worker_device)
        self.networks.append(self.policy_network)

        # signals definition
        self.value_loss = Signal('Value Loss')
        self.signals.append(self.value_loss)
        self.policy_loss = Signal('Policy Loss')
        self.signals.append(self.policy_loss)
        self.kl_divergence = Signal('KL Divergence')
        self.signals.append(self.kl_divergence)
        self.total_kl_divergence_during_training_process = 0.0
        self.unclipped_grads = Signal('Grads (unclipped)')
        self.signals.append(self.unclipped_grads)

        self.reset_game(do_not_reset_env=True)

    def fill_advantages(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)

        # * Found not to have any impact *
        # current_states_with_timestep = self.concat_state_and_timestep(batch)

        current_state_values = self.critic_network.online_network.predict(current_states).squeeze()

        # calculate advantages
        advantages = []
        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            advantages = total_return - current_state_values
        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            episode_start_idx = 0
            advantages = np.array([])
            # current_state_values[game_overs] = 0
            for idx, game_over in enumerate(game_overs):
                if game_over:
                    # get advantages for the rollout
                    value_bootstrapping = np.zeros((1,))
                    rollout_state_values = np.append(current_state_values[episode_start_idx:idx+1], value_bootstrapping)

                    rollout_advantages, _ = \
                        self.get_general_advantage_estimation_values(rewards[episode_start_idx:idx+1],
                                                                     rollout_state_values)
                    episode_start_idx = idx + 1
                    advantages = np.append(advantages, rollout_advantages)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # standardize
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        for transition, advantage in zip(self.memory.transitions, advantages):
            transition.info['advantage'] = advantage

        self.action_advantages.add_sample(advantages)

    def train_value_network(self, dataset, epochs):
        loss = []
        current_states, _, _, _, _, total_return = self.extract_batch(dataset)

        # * Found not to have any impact *
        # add a timestep to the observation
        # current_states_with_timestep = self.concat_state_and_timestep(dataset)

        total_return = np.expand_dims(total_return, -1)
        mix_fraction = self.tp.agent.value_targets_mix_fraction
        for j in range(epochs):
            batch_size = len(dataset)
            if self.critic_network.online_network.optimizer_type != 'LBFGS':
                batch_size = self.tp.batch_size
            for i in range(len(dataset) // batch_size):
                # split to batches for first order optimization techniques
                current_states_batch = {
                    k: v[i * batch_size:(i + 1) * batch_size]
                    for k, v in current_states.items()
                }
                total_return_batch = total_return[i * batch_size:(i + 1) * batch_size]
                old_policy_values = force_list(self.critic_network.target_network.predict(
                    current_states_batch).squeeze())
                if self.critic_network.online_network.optimizer_type != 'LBFGS':
                    targets = total_return_batch
                else:
                    current_values = self.critic_network.online_network.predict(current_states_batch)
                    targets = current_values * (1 - mix_fraction) + total_return_batch * mix_fraction

                inputs = copy.copy(current_states_batch)
                for input_index, input in enumerate(old_policy_values):
                    name = 'output_0_{}'.format(input_index)
                    if name in self.critic_network.online_network.inputs:
                        inputs[name] = input

                value_loss = self.critic_network.online_network.accumulate_gradients(inputs, targets)
                self.critic_network.apply_gradients_to_online_network()
                if self.tp.distributed:
                    self.critic_network.apply_gradients_to_global_network()
                self.critic_network.online_network.reset_accumulated_gradients()

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
            for i in range(len(dataset) // self.tp.batch_size):
                batch = dataset[i * self.tp.batch_size:(i + 1) * self.tp.batch_size]
                current_states, _, actions, _, _, total_return = self.extract_batch(batch)
                advantages = np.array([t.info['advantage'] for t in batch])
                if not self.tp.env_instance.discrete_controls and len(actions.shape) == 1:
                    actions = np.expand_dims(actions, -1)

                # get old policy probabilities and distribution
                old_policy = force_list(self.policy_network.target_network.predict(current_states))

                # calculate gradients and apply on both the local policy network and on the global policy network
                fetches = [self.policy_network.online_network.output_heads[0].kl_divergence,
                           self.policy_network.online_network.output_heads[0].entropy]

                inputs = copy.copy(current_states)
                # TODO: why is this output 0 and not output 1?
                inputs['output_0_0'] = actions
                # TODO: does old_policy_distribution really need to be represented as a list?
                # A: yes it does, in the event of discrete controls, it has just a mean
                # otherwise, it has both a mean and standard deviation
                for input_index, input in enumerate(old_policy):
                    inputs['output_0_{}'.format(input_index + 1)] = input
                total_loss, policy_losses, unclipped_grads, fetch_result =\
                    self.policy_network.online_network.accumulate_gradients(
                        inputs, [advantages], additional_fetches=fetches)

                self.policy_network.apply_gradients_to_online_network()
                if self.tp.distributed:
                    self.policy_network.apply_gradients_to_global_network()

                self.policy_network.online_network.reset_accumulated_gradients()

                loss['total_loss'].append(total_loss)
                loss['policy_losses'].append(policy_losses)
                loss['unclipped_grads'].append(unclipped_grads)
                loss['fetch_result'].append(fetch_result)

                self.unclipped_grads.add_sample(unclipped_grads)

            for key in loss.keys():
                loss[key] = np.mean(loss[key], 0)

            if self.tp.learning_rate_decay_rate != 0:
                curr_learning_rate = self.main_network.online_network.get_variable_value(self.tp.learning_rate)
                self.curr_learning_rate.add_sample(curr_learning_rate)
            else:
                curr_learning_rate = self.tp.learning_rate

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
        kl_target = self.tp.agent.target_kl_divergence
        kl_coefficient = self.policy_network.online_network.get_variable_value(
            self.policy_network.online_network.output_heads[0].kl_coefficient)
        new_kl_coefficient = kl_coefficient
        if self.total_kl_divergence_during_training_process > 1.3 * kl_target:
            # kl too high => increase regularization
            new_kl_coefficient *= 1.5
        elif self.total_kl_divergence_during_training_process < 0.7 * kl_target:
            # kl too low => decrease regularization
            new_kl_coefficient /= 1.5

        # update the kl coefficient variable
        if kl_coefficient != new_kl_coefficient:
            self.policy_network.online_network.set_variable_value(
                self.policy_network.online_network.output_heads[0].assign_kl_coefficient,
                new_kl_coefficient,
                self.policy_network.online_network.output_heads[0].kl_coefficient_ph)

        screen.log_title("KL penalty coefficient change = {} -> {}".format(kl_coefficient, new_kl_coefficient))

    def post_training_commands(self):
        if self.tp.agent.use_kl_regularization:
            self.update_kl_coefficient()

        # clean memory
        self.memory.clean()

    def train(self):
        self.policy_network.sync()
        self.critic_network.sync()

        dataset = self.memory.transitions

        self.fill_advantages(dataset)

        # take only the requested number of steps
        dataset = dataset[:self.tp.agent.num_consecutive_playing_steps]

        value_loss = self.train_value_network(dataset, 1)
        policy_loss = self.train_policy_network(dataset, 10)

        self.value_loss.add_sample(value_loss)
        self.policy_loss.add_sample(policy_loss)
        self.update_log()  # should be done in order to update the data that has been accumulated * while not playing *
        return np.append(value_loss, policy_loss)

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        if self.env.discrete_controls:
            # DISCRETE
            action_values = self.policy_network.online_network.predict(self.tf_input_state(curr_state)).squeeze()

            if phase == RunPhase.TRAIN:
                action = self.exploration_policy.get_action(action_values)
            else:
                action = np.argmax(action_values)
            action_info = {"action_probability": action_values[action]}
            # self.entropy.add_sample(-np.sum(action_values * np.log(action_values)))
        else:
            # CONTINUOUS
            action_values_mean, action_values_std = self.policy_network.online_network.predict(self.tf_input_state(curr_state))
            action_values_mean = action_values_mean.squeeze()
            action_values_std = action_values_std.squeeze()
            if phase == RunPhase.TRAIN:
                action = np.squeeze(np.random.randn(1, self.action_space_size) * action_values_std + action_values_mean)
            else:
                action = action_values_mean
            action_info = {"action_probability": action_values_mean}

        return action, action_info
