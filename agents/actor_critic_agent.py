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

from agents.policy_optimization_agent import *
from logger import *
from utils import *
import scipy.signal


# Actor Critic - https://arxiv.org/abs/1602.01783
class ActorCriticAgent(PolicyOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0, create_target_network = False):
        PolicyOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id, create_target_network)
        self.last_gradient_update_step_idx = 0
        self.action_advantages = Signal('Advantages')
        self.state_values = Signal('Values')
        self.unclipped_grads = Signal('Grads (unclipped)')
        self.signals.append(self.action_advantages)
        self.signals.append(self.state_values)
        self.signals.append(self.unclipped_grads)

    # Discounting function used to calculate discounted returns.
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def get_general_advantage_estimation_values(self, rewards, values):
        # values contain n+1 elements (t ... t+n+1), rewards contain n elements (t ... t + n)
        bootstrap_extended_rewards = np.array(rewards.tolist() + [values[-1]])

        # Approximation based calculation of GAE (mathematically correct only when Tmax = inf,
        # although in practice works even in much smaller Tmax values, e.g. 20)
        deltas = rewards + self.tp.agent.discount * values[1:] - values[:-1]
        gae = self.discount(deltas, self.tp.agent.discount * self.tp.agent.gae_lambda)

        if self.tp.agent.estimate_value_using_gae:
            discounted_returns = np.expand_dims(gae + values[:-1], -1)
        else:
            discounted_returns = np.expand_dims(np.array(self.discount(bootstrap_extended_rewards,
                                                                       self.tp.agent.discount)), 1)[:-1]
        return gae, discounted_returns

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # get the values for the current states
        result = self.main_network.online_network.predict(current_states)
        current_state_values = result[0]
        self.state_values.add_sample(current_state_values)

        # the targets for the state value estimator
        num_transitions = len(game_overs)
        state_value_head_targets = np.zeros((num_transitions, 1))

        # estimate the advantage function
        action_advantages = np.zeros((num_transitions, 1))

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            if game_overs[-1]:
                R = 0
            else:
                R = self.main_network.online_network.predict(np.expand_dims(next_states[-1], 0))[0]

            for i in reversed(range(num_transitions)):
                R = rewards[i] + self.tp.agent.discount * R
                state_value_head_targets[i] = R
                action_advantages[i] = R - current_state_values[i]

        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            bootstrapped_value = self.main_network.online_network.predict(np.expand_dims(next_states[-1], 0))[0]
            values = np.append(current_state_values, bootstrapped_value)
            if game_overs[-1]:
                values[-1] = 0

            # get general discounted returns table
            gae_values, state_value_head_targets = self.get_general_advantage_estimation_values(rewards, values)
            action_advantages = np.vstack(gae_values)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        action_advantages = action_advantages.squeeze(axis=-1)
        if not self.env.discrete_controls and len(actions.shape) < 2:
            actions = np.expand_dims(actions, -1)

        # train
        result = self.main_network.online_network.accumulate_gradients([current_states, actions],
                                                                       [state_value_head_targets, action_advantages])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        logger.create_signal_value('Value Loss', losses[0])
        logger.create_signal_value('Policy Loss', losses[1])

        return total_loss

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        # convert to batch so we can run it through the network
        observation = np.expand_dims(np.array(curr_state['observation']), 0)
        if self.env.discrete_controls:
            # DISCRETE
            state_value, action_probabilities = self.main_network.online_network.predict(observation)
            action_probabilities = action_probabilities.squeeze()
            if phase == RunPhase.TRAIN:
                action = self.exploration_policy.get_action(action_probabilities)
            else:
                action = np.argmax(action_probabilities)
            action_info = {"action_probability": action_probabilities[action], "state_value": state_value}
            self.entropy.add_sample(-np.sum(action_probabilities * np.log(action_probabilities + eps)))
        else:
            # CONTINUOUS
            state_value, action_values_mean, action_values_std = self.main_network.online_network.predict(observation)
            action_values_mean = action_values_mean.squeeze()
            action_values_std = action_values_std.squeeze()
            if phase == RunPhase.TRAIN:
                action = np.squeeze(np.random.randn(1, self.action_space_size) * action_values_std + action_values_mean)
            else:
                action = action_values_mean
            action_info = {"action_probability": action, "state_value": state_value}

        return action, action_info
