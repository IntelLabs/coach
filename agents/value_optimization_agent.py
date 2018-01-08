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

from agents.agent import *


class ValueOptimizationAgent(Agent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0, create_target_network=True):
        Agent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.main_network = NetworkWrapper(tuning_parameters, create_target_network, self.has_global, 'main',
                                           self.replicated_device, self.worker_device)
        self.networks.append(self.main_network)
        self.q_values = Signal("Q")
        self.signals.append(self.q_values)

    # Algorithms for which q_values are calculated from predictions will override this function
    def get_q_values(self, prediction):
        return prediction

    def tf_input_state(self, curr_state):
        """
        convert curr_state into input tensors tensorflow is expecting.

        TODO: move this function into Agent and use in as many agent implementations as possible
        currently, other agents will likely not work with environment measurements.
        This will become even more important as we support more complex and varied environment states.
        """
        # convert to batch so we can run it through the network
        observation = np.expand_dims(np.array(curr_state['observation']), 0)
        if self.tp.agent.use_measurements:
            measurements = np.expand_dims(np.array(curr_state['measurements']), 0)
            tf_input_state = [observation, measurements]
        else:
            tf_input_state = observation
        return tf_input_state

    def get_prediction(self, curr_state):
        return self.main_network.online_network.predict(self.tf_input_state(curr_state))

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        prediction = self.get_prediction(curr_state)
        actions_q_values = self.get_q_values(prediction)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        if phase == RunPhase.TRAIN:
            action = self.exploration_policy.get_action(actions_q_values)
        else:
            action = self.evaluation_exploration_policy.get_action(actions_q_values)

        # this is for bootstrapped dqn
        if type(actions_q_values) == list and len(actions_q_values) > 0:
            actions_q_values = actions_q_values[self.exploration_policy.selected_head]
        actions_q_values = actions_q_values.squeeze()

        # store the q values statistics for logging
        self.q_values.add_sample(actions_q_values)

        # store information for plotting interactively (actual plotting is done in agent)
        if self.tp.visualization.plot_action_values_online:
            for idx, action_name in enumerate(self.env.actions_description):
                self.episode_running_info[action_name].append(actions_q_values[idx])

        action_value = {"action_value": actions_q_values[action]}
        return action, action_value
