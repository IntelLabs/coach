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

# Based on on the description in:
# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

# Ornstein-Uhlenbeck process
class OUProcess(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.action_space_size = tuning_parameters.env.action_space_size
        self.mu = float(tuning_parameters.exploration.mu) * np.ones(self.action_space_size)
        self.theta = tuning_parameters.exploration.theta
        self.sigma = float(tuning_parameters.exploration.sigma) * np.ones(self.action_space_size)
        self.state = np.zeros(self.action_space_size)
        self.dt = tuning_parameters.exploration.dt

    def reset(self):
        self.state = np.zeros(self.action_space_size)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state[0]

    def get_action(self, action_values):
        noise = self.noise()
        return action_values.squeeze() + noise

    def get_control_param(self):
        return self.state[0]
