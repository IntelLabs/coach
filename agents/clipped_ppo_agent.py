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


# Clipped Proximal Policy Optimization - https://arxiv.org/abs/1707.06347
class ClippedPPOAgent(ActorCriticAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ActorCriticAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id,
                                  create_target_network=True)
        # signals definition
        self.value_loss = Signal('Value Loss')
        self.signals.append(self.value_loss)
        self.policy_loss = Signal('Policy Loss')
        self.signals.append(self.policy_loss)
        self.total_kl_divergence_during_training_process = 0.0
        self.unclipped_grads = Signal('Grads (unclipped)')
        self.signals.append(self.unclipped_grads)
        self.value_targets = Signal('Value Targets')
        self.signals.append(self.value_targets)
        self.kl_divergence = Signal('KL Divergence')
        self.signals.append(self.kl_divergence)

    def fill_advantages(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)

        current_state_values = self.main_network.online_network.predict(current_states)[0]
        current_state_values = current_state_values.squeeze()
        self.state_values.add_sample(current_state_values)

        # calculate advantages
        advantages = []
        value_targets = []
        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            advantages = total_return - current_state_values
        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            episode_start_idx = 0
            advantages = np.array([])
            value_targets = np.array([])
            for idx, game_over in enumerate(game_overs):
                if game_over:
                    # get advantages for the rollout
                    value_bootstrapping = np.zeros((1,))
                    rollout_state_values = np.append(current_state_values[episode_start_idx:idx+1], value_bootstrapping)

                    rollout_advantages, gae_based_value_targets = \
                        self.get_general_advantage_estimation_values(rewards[episode_start_idx:idx+1],
                                                                     rollout_state_values)
                    episode_start_idx = idx + 1
                    advantages = np.append(advantages, rollout_advantages)
                    value_targets = np.append(value_targets, gae_based_value_targets)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # standardize
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for transition, advantage, value_target in zip(batch, advantages, value_targets):
            transition.info['advantage'] = advantage
            transition.info['gae_based_value_target'] = value_target

        self.action_advantages.add_sample(advantages)

    def train_network(self, dataset, epochs):
        loss = []
        for j in range(epochs):
            loss = {
                'total_loss': [],
                'policy_losses': [],
                'unclipped_grads': [],
                'fetch_result': []
            }
            shuffle(dataset)
            for i in range(int(len(dataset) / self.tp.batch_size)):
                batch = dataset[i * self.tp.batch_size:(i + 1) * self.tp.batch_size]
                current_states, _, actions, _, _, total_return = self.extract_batch(batch)

                advantages = np.array([t.info['advantage'] for t in batch])
                gae_based_value_targets = np.array([t.info['gae_based_value_target'] for t in batch])
                if not self.tp.env_instance.discrete_controls and len(actions.shape) == 1:
                    actions = np.expand_dims(actions, -1)

                # get old policy probabilities and distribution
                result = self.main_network.target_network.predict(current_states)
                old_policy_distribution = result[1:]

                # calculate gradients and apply on both the local policy network and on the global policy network
                fetches = [self.main_network.online_network.output_heads[1].kl_divergence,
                           self.main_network.online_network.output_heads[1].entropy]

                total_return = np.expand_dims(total_return, -1)
                value_targets = gae_based_value_targets if self.tp.agent.estimate_value_using_gae else total_return
                inputs = copy.copy(current_states)
                # TODO: why is this output 0 and not output 1?
                inputs['output_0_0'] = actions
                # TODO: does old_policy_distribution really need to be represented as a list?
                # A: yes it does, in the event of discrete controls, it has just a mean
                # otherwise, it has both a mean and standard deviation
                for input_index, input in enumerate(old_policy_distribution):
                    inputs['output_0_{}'.format(input_index + 1)] = input
                total_loss, policy_losses, unclipped_grads, fetch_result =\
                    self.main_network.online_network.accumulate_gradients(
                        inputs, [total_return, advantages], additional_fetches=fetches)

                self.value_targets.add_sample(value_targets)
                if self.tp.distributed:
                    self.main_network.apply_gradients_to_global_network()
                    self.main_network.update_online_network()
                else:
                    self.main_network.apply_gradients_to_online_network()

                self.main_network.online_network.reset_accumulated_gradients()

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
        return policy_losses

    def post_training_commands(self):

        # clean memory
        self.memory.clean()

    def train(self):
        self.main_network.sync()

        dataset = self.memory.transitions

        self.fill_advantages(dataset)

        # take only the requested number of steps
        dataset = dataset[:self.tp.agent.num_consecutive_playing_steps]

        if self.tp.distributed and self.tp.agent.share_statistics_between_workers:
            self.running_observation_stats.push(np.array([np.array(t.state['observation']) for t in dataset]))

        losses = self.train_network(dataset, 10)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])
        self.update_log()  # should be done in order to update the data that has been accumulated * while not playing *
        return np.append(losses[0], losses[1])

    def choose_action(self, current_state, phase=RunPhase.TRAIN):
        if self.env.discrete_controls:
            # DISCRETE
            _, action_values = self.main_network.online_network.predict(self.tf_input_state(current_state))
            action_values = action_values.squeeze()

            if phase == RunPhase.TRAIN:
                action = self.exploration_policy.get_action(action_values)
            else:
                action = np.argmax(action_values)
            action_info = {"action_probability": action_values[action]}
            # self.entropy.add_sample(-np.sum(action_values * np.log(action_values)))
        else:
            # CONTINUOUS
            _, action_values_mean, action_values_std = self.main_network.online_network.predict(self.tf_input_state(current_state))
            action_values_mean = action_values_mean.squeeze()
            action_values_std = action_values_std.squeeze()
            if phase == RunPhase.TRAIN:
                action = np.squeeze(np.random.randn(1, self.action_space_size) * action_values_std + action_values_mean)
                # if self.current_episode % 5 == 0 and self.current_episode_steps_counter < 5:
                #     print action
            else:
                action = action_values_mean
            action_info = {"action_probability": action_values_mean}

        return action, action_info
