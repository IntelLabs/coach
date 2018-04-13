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
import numpy as np

from agents import value_optimization_agent as voa
import utils

# Bootstrapped DQN - https://arxiv.org/pdf/1602.04621.pdf
class BootstrappedDQNAgent(voa.ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        voa.ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)

    def reset_game(self, do_not_reset_env=False):
        voa.ValueOptimizationAgent.reset_game(self, do_not_reset_env)
        self.exploration_policy.select_head()

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # for the action we actually took, the error is:
        # TD error = r + discount*max(q_st_plus_1) - q_st
        # for all other actions, the error is 0
        q_st_plus_1 = self.main_network.target_network.predict(next_states)
        # initialize with the current prediction so that we will
        TD_targets = self.main_network.online_network.predict(current_states)

        #  only update the action that we have actually done in this transition
        for i in range(self.tp.batch_size):
            mask = batch[i].info['mask']
            for head_idx in range(self.tp.exploration.architecture_num_q_heads):
                if mask[head_idx] == 1:
                    TD_targets[head_idx][i, actions[i]] = rewards[i] + \
                                                          (1.0 - game_overs[i]) * self.tp.agent.discount * np.max(
                                                              q_st_plus_1[head_idx][i], 0)

        result = self.main_network.train_and_sync_networks(current_states, TD_targets)

        total_loss = result[0]

        return total_loss

    def act(self, phase=utils.RunPhase.TRAIN):
        voa.ValueOptimizationAgent.act(self, phase)
        mask = np.random.binomial(1, self.tp.exploration.bootstrapped_data_sharing_probability,
                                  self.tp.exploration.architecture_num_q_heads)
        self.memory.update_last_transition_info({'mask': mask})
