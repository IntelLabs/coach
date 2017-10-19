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
from exploration_policies.exploration_policy import *


class AdditiveNoise(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.variance = tuning_parameters.exploration.initial_noise_variance_percentage
        self.final_variance = tuning_parameters.exploration.final_noise_variance_percentage
        self.decay_steps = tuning_parameters.exploration.noise_variance_decay_steps
        self.variance_decay_delta = (self.variance - self.final_variance) / float(self.decay_steps)

    def decay_exploration(self):
        if self.variance > self.final_variance:
            self.variance -= self.variance_decay_delta
        elif self.variance < self.final_variance:
            self.variance = self.final_variance

    def get_action(self, action_values):
        if self.phase == RunPhase.TRAIN:
            self.decay_exploration()
        action = np.random.normal(action_values, 2 * self.variance * self.action_abs_range)
        return action #np.clip(action, -self.action_abs_range, self.action_abs_range).squeeze()

    def get_control_param(self):
        return self.variance
