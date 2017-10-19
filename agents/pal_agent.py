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


# Persistent Advantage Learning - https://arxiv.org/pdf/1512.04860.pdf
class PALAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.alpha = tuning_parameters.agent.pal_alpha
        self.persistent = tuning_parameters.agent.persistent_advantage_learning
        self.monte_carlo_mixing_rate = tuning_parameters.agent.monte_carlo_mixing_rate

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)

        selected_actions = np.argmax(self.main_network.online_network.predict(next_states), 1)

        # next state values
        q_st_plus_1_target = self.main_network.target_network.predict(next_states)
        v_st_plus_1_target = np.max(q_st_plus_1_target, 1)

        # current state values according to online network
        q_st_online = self.main_network.online_network.predict(current_states)

        # current state values according to target network
        q_st_target = self.main_network.target_network.predict(current_states)
        v_st_target = np.max(q_st_target, 1)

        # calculate TD error
        TD_targets = np.copy(q_st_online)
        for i in range(self.tp.batch_size):
            TD_targets[i, actions[i]] = rewards[i] + (1.0 - game_overs[i]) * self.tp.agent.discount * \
                                                     q_st_plus_1_target[i][selected_actions[i]]
            advantage_learning_update = v_st_target[i] - q_st_target[i, actions[i]]
            next_advantage_learning_update = v_st_plus_1_target[i] - q_st_plus_1_target[i, selected_actions[i]]
            # Persistent Advantage Learning or Regular Advantage Learning
            if self.persistent:
                TD_targets[i, actions[i]] -= self.alpha * min(advantage_learning_update, next_advantage_learning_update)
            else:
                TD_targets[i, actions[i]] -= self.alpha * advantage_learning_update

            # mixing monte carlo updates
            monte_carlo_target = total_return[i]
            TD_targets[i, actions[i]] = (1 - self.monte_carlo_mixing_rate) * TD_targets[i, actions[i]] \
                                        + self.monte_carlo_mixing_rate * monte_carlo_target

        result = self.main_network.train_and_sync_networks(current_states, TD_targets)
        total_loss = result[0]

        return total_loss
