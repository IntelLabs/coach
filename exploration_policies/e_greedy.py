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


class EGreedy(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.epsilon = tuning_parameters.exploration.initial_epsilon
        self.final_epsilon = tuning_parameters.exploration.final_epsilon
        self.epsilon_decay_delta = (
                                   tuning_parameters.exploration.initial_epsilon - tuning_parameters.exploration.final_epsilon) \
                                   / float(tuning_parameters.exploration.epsilon_decay_steps)
        self.evaluation_epsilon = tuning_parameters.exploration.evaluation_epsilon

        # for continuous e-greedy (see http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)
        self.variance = tuning_parameters.exploration.initial_noise_variance_percentage
        self.final_variance = tuning_parameters.exploration.final_noise_variance_percentage
        self.decay_steps = tuning_parameters.exploration.noise_variance_decay_steps
        self.variance_decay_delta = (self.variance - self.final_variance) / float(self.decay_steps)

    def decay_exploration(self):
        # decay epsilon
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay_delta
        elif self.epsilon < self.final_epsilon:
            self.epsilon = self.final_epsilon

        # decay noise variance
        if not self.discrete_controls:
            if self.variance > self.final_variance:
                self.variance -= self.variance_decay_delta
            elif self.variance < self.final_variance:
                self.variance = self.final_variance

    def get_action(self, action_values):
        if self.phase == RunPhase.TRAIN:
            self.decay_exploration()
        epsilon = self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon

        if self.discrete_controls:
            top_action = np.argmax(action_values)
            if np.random.rand() < epsilon:
                return np.random.randint(self.action_space_size)
            else:
                return top_action
        else:
            noise = np.random.randn(1, self.action_space_size) * self.variance * self.action_abs_range
            return np.squeeze(action_values + (np.random.rand() < epsilon) * noise)

    def get_control_param(self):
        return self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon
