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


# Distributional Deep Q Network - https://arxiv.org/pdf/1707.06887.pdf
class DistributionalDQNAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.z_values = np.linspace(self.tp.agent.v_min, self.tp.agent.v_max, self.tp.agent.atoms)

    # prediction's format is (batch,actions,atoms)
    def get_q_values(self, prediction):
        return np.dot(prediction, self.z_values)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # for the action we actually took, the error is calculated by the atoms distribution
        # for all other actions, the error is 0
        distributed_q_st_plus_1 = self.main_network.target_network.predict(next_states)
        # initialize with the current prediction so that we will
        TD_targets = self.main_network.online_network.predict(current_states)

        # only update the action that we have actually done in this transition
        target_actions = np.argmax(self.get_q_values(distributed_q_st_plus_1), axis=1)
        m = np.zeros((self.tp.batch_size, self.z_values.size))

        batches = np.arange(self.tp.batch_size)
        for j in range(self.z_values.size):
            tzj = np.fmax(np.fmin(rewards + (1.0 - game_overs) * self.tp.agent.discount * self.z_values[j],
                                self.z_values[self.z_values.size - 1]),
                                self.z_values[0])
            bj = (tzj - self.z_values[0])/(self.z_values[1] - self.z_values[0])
            u = (np.ceil(bj)).astype(int)
            l = (np.floor(bj)).astype(int)
            m[batches, l] = m[batches, l] + (distributed_q_st_plus_1[batches, target_actions, j] * (u - bj))
            m[batches, u] = m[batches, u] + (distributed_q_st_plus_1[batches, target_actions, j] * (bj - l))
        # total_loss = cross entropy between actual result above and predicted result for the given action
        TD_targets[batches, actions] = m

        result = self.main_network.train_and_sync_networks(current_states, TD_targets)
        total_loss = result[0]

        return total_loss

