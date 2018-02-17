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
import scipy.signal

from agents.value_optimization_agent import ValueOptimizationAgent
from agents.policy_optimization_agent import PolicyOptimizationAgent
from logger import logger
from utils import Signal, last_sample


# N Step Q Learning Agent - https://arxiv.org/abs/1602.01783
class NStepQAgent(ValueOptimizationAgent, PolicyOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id, create_target_network=True)
        self.last_gradient_update_step_idx = 0
        self.q_values = Signal('Q Values')
        self.unclipped_grads = Signal('Grads (unclipped)')
        self.value_loss = Signal('Value Loss')
        self.signals.append(self.q_values)
        self.signals.append(self.unclipped_grads)
        self.signals.append(self.value_loss)

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # get the values for the current states
        state_value_head_targets = self.main_network.online_network.predict(current_states)

        # the targets for the state value estimator
        num_transitions = len(game_overs)

        if self.tp.agent.targets_horizon == '1-Step':
            # 1-Step Q learning
            q_st_plus_1 = self.main_network.target_network.predict(next_states)

            for i in reversed(range(num_transitions)):
                state_value_head_targets[i][actions[i]] = \
                    rewards[i] + (1.0 - game_overs[i]) * self.tp.agent.discount * np.max(q_st_plus_1[i], 0)

        elif self.tp.agent.targets_horizon == 'N-Step':
            # N-Step Q learning
            if game_overs[-1]:
                R = 0
            else:
                R = np.max(self.main_network.target_network.predict(last_sample(next_states)))

            for i in reversed(range(num_transitions)):
                R = rewards[i] + self.tp.agent.discount * R
                state_value_head_targets[i][actions[i]] = R

        else:
            assert True, 'The available values for targets_horizon are: 1-Step, N-Step'

        # train
        result = self.main_network.online_network.accumulate_gradients(current_states, [state_value_head_targets])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])

        return total_loss

    def train(self):
        # update the target network of every network that has a target network
        if self.total_steps_counter % self.tp.agent.num_steps_between_copying_online_weights_to_target == 0:
            for network in self.networks:
                network.update_target_network(self.tp.agent.rate_for_copying_weights_to_target)
            logger.create_signal_value('Update Target Network', 1)
        else:
            logger.create_signal_value('Update Target Network', 0, overwrite=False)

        return PolicyOptimizationAgent.train(self)
