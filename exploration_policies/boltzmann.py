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

from exploration_policies.exploration_policy import *


class Boltzmann(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.temperature = tuning_parameters.exploration.initial_temperature
        self.final_temperature = tuning_parameters.exploration.final_temperature
        self.temperature_decay_delta = (
                                       tuning_parameters.exploration.initial_temperature - tuning_parameters.exploration.final_temperature) \
                                       / float(tuning_parameters.exploration.temperature_decay_steps)

    def decay_temperature(self):
        if self.temperature > self.final_temperature:
            self.temperature -= self.temperature_decay_delta

    def get_action(self, action_values):
        if self.phase == RunPhase.TRAIN:
            self.decay_temperature()
        # softmax calculation
        exp_probabilities = np.exp(action_values / self.temperature)
        probabilities = exp_probabilities / np.sum(exp_probabilities)
        probabilities[-1] = 1 - np.sum(probabilities[:-1])  # make sure probs sum to 1
        # choose actions according to the probabilities
        return np.random.choice(range(self.action_space_size), p=probabilities)

    def get_control_param(self):
        return self.temperature
