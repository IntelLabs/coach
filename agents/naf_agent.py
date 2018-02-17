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

from agents.value_optimization_agent import ValueOptimizationAgent
from utils import RunPhase, Signal


# Normalized Advantage Functions - https://arxiv.org/pdf/1603.00748.pdf
class NAFAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.l_values = Signal("L")
        self.a_values = Signal("Advantage")
        self.mu_values = Signal("Action")
        self.v_values = Signal("V")
        self.signals += [self.l_values, self.a_values, self.mu_values, self.v_values]

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # TD error = r + discount*v_st_plus_1 - q_st
        v_st_plus_1 = self.main_network.target_network.predict(
            next_states,
            self.main_network.target_network.output_heads[0].V,
            squeeze_output=False,
        )
        TD_targets = np.expand_dims(rewards, -1) + (1.0 - np.expand_dims(game_overs, -1)) * self.tp.agent.discount * v_st_plus_1

        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, -1)

        result = self.main_network.train_and_sync_networks({**current_states, 'output_0_0': actions}, TD_targets)
        total_loss = result[0]

        return total_loss

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        assert not self.env.discrete_controls, 'NAF works only for continuous control problems'

        # convert to batch so we can run it through the network
        # observation = np.expand_dims(np.array(curr_state['observation']), 0)
        naf_head = self.main_network.online_network.output_heads[0]
        action_values = self.main_network.online_network.predict(
            self.tf_input_state(curr_state),
            outputs=naf_head.mu,
            squeeze_output=False,
        )
        if phase == RunPhase.TRAIN:
            action = self.exploration_policy.get_action(action_values)
        else:
            action = action_values

        Q, L, A, mu, V = self.main_network.online_network.predict(
            {**self.tf_input_state(curr_state), 'output_0_0': action_values},
            outputs=[naf_head.Q, naf_head.L, naf_head.A, naf_head.mu, naf_head.V],
        )

        # store the q values statistics for logging
        self.q_values.add_sample(Q)
        self.l_values.add_sample(L)
        self.a_values.add_sample(A)
        self.mu_values.add_sample(mu)
        self.v_values.add_sample(V)

        action_value = {"action_value": Q}
        return action, action_value
