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

from agents.agent import *


# Imitation Agent
class ImitationAgent(Agent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        Agent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.main_network = NetworkWrapper(tuning_parameters, False, self.has_global, 'main',
                                           self.replicated_device, self.worker_device)
        self.networks.append(self.main_network)
        self.imitation = True

    def extract_action_values(self, prediction):
        return prediction.squeeze()

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        # convert to batch so we can run it through the network
        prediction = self.main_network.online_network.predict(self.tf_input_state(curr_state))

        # get action values and extract the best action from it
        action_values = self.extract_action_values(prediction)
        if self.env.discrete_controls:
            # DISCRETE
            # action = np.argmax(action_values)
            action = self.evaluation_exploration_policy.get_action(action_values)
            action_value = {"action_probability": action_values[action]}
        else:
            # CONTINUOUS
            action = action_values
            action_value = {}

        return action, action_value

    def log_to_screen(self, phase):
        # log to screen
        if phase == RunPhase.TRAIN:
            # for the training phase - we log during the episode to visualize the progress in training
            screen.log_dict(
                OrderedDict([
                    ("Worker", self.task_id),
                    ("Episode", self.current_episode),
                    ("Loss", self.loss.values[-1]),
                    ("Training iteration", self.training_iteration)
                ]),
                prefix="Training"
            )
        else:
            # for the evaluation phase - logging as in regular RL
            Agent.log_to_screen(self, phase)
