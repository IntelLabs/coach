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


class MixedMonteCarloAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.mixing_rate = tuning_parameters.agent.monte_carlo_mixing_rate

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)

        TD_targets = self.main_network.online_network.predict(current_states)
        selected_actions = np.argmax(self.main_network.online_network.predict(next_states), 1)
        q_st_plus_1 = self.main_network.target_network.predict(next_states)
        # initialize with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        for i in range(self.tp.batch_size):
            one_step_target = rewards[i] + (1.0 - game_overs[i]) * self.tp.agent.discount * q_st_plus_1[i][
                selected_actions[i]]
            monte_carlo_target = total_return[i]
            TD_targets[i, actions[i]] = (1 - self.mixing_rate) * one_step_target + self.mixing_rate * monte_carlo_target

        result = self.main_network.train_and_sync_networks(current_states, TD_targets)
        total_loss = result[0]

        return total_loss
