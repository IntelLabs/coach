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
from utils import *
from configurations import *


class ExplorationPolicy(object):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        self.phase = RunPhase.HEATUP
        self.action_space_size = tuning_parameters.env.action_space_size
        self.action_abs_range = tuning_parameters.env_instance.action_space_abs_range
        self.discrete_controls = tuning_parameters.env_instance.discrete_controls

    def reset(self):
        """
        Used for resetting the exploration policy parameters when needed
        :return: None
        """
        pass

    def get_action(self, action_values):
        """
        Given a list of values corresponding to each action, 
        choose one actions according to the exploration policy
        :param action_values: A list of action values
        :return: The chosen action
        """
        pass

    def change_phase(self, phase):
        """
        Change between running phases of the algorithm
        :param phase: Either Heatup or Train
        :return: none
        """
        self.phase = phase

    def get_control_param(self):
        return 0