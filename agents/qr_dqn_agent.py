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

from agents.value_optimization_agent import *


# Quantile Regression Deep Q Network - https://arxiv.org/pdf/1710.10044v1.pdf
class QuantileRegressionDQNAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.quantile_probabilities = np.ones(self.tp.agent.atoms) / float(self.tp.agent.atoms)

    # prediction's format is (batch,actions,atoms)
    def get_q_values(self, quantile_values):
        return np.dot(quantile_values, self.quantile_probabilities)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # get the quantiles of the next states and current states
        next_state_quantiles = self.main_network.target_network.predict(next_states)
        current_quantiles = self.main_network.online_network.predict(current_states)

        # get the optimal actions to take for the next states
        target_actions = np.argmax(self.get_q_values(next_state_quantiles), axis=1)

        # calculate the Bellman update
        batch_idx = list(range(self.tp.batch_size))
        rewards = np.expand_dims(rewards, -1)
        game_overs = np.expand_dims(game_overs, -1)
        TD_targets = rewards + (1.0 - game_overs) * self.tp.agent.discount \
                               * next_state_quantiles[batch_idx, target_actions]

        # get the locations of the selected actions within the batch for indexing purposes
        actions_locations = [[b, a] for b, a in zip(batch_idx, actions)]

        # calculate the cumulative quantile probabilities and reorder them to fit the sorted quantiles order
        cumulative_probabilities = np.array(range(self.tp.agent.atoms+1))/float(self.tp.agent.atoms)  # tau_i
        quantile_midpoints = 0.5*(cumulative_probabilities[1:] + cumulative_probabilities[:-1])  # tau^hat_i
        quantile_midpoints = np.tile(quantile_midpoints, (self.tp.batch_size, 1))
        sorted_quantiles = np.argsort(current_quantiles[batch_idx, actions])
        for idx in range(self.tp.batch_size):
            quantile_midpoints[idx, :] = quantile_midpoints[idx, sorted_quantiles[idx]]

        # train
        result = self.main_network.train_and_sync_networks({
            **current_states,
            'output_0_0': actions_locations,
            'output_0_1': quantile_midpoints,
        }, TD_targets)
        total_loss = result[0]

        return total_loss
