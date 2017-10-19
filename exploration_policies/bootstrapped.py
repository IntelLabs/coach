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

from exploration_policies.e_greedy import *


class Bootstrapped(EGreedy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running parameters
        :type tuning_parameters: Preset
        """
        EGreedy.__init__(self, tuning_parameters)
        self.num_heads = tuning_parameters.exploration.architecture_num_q_heads
        self.selected_head = 0

    def select_head(self):
        self.selected_head = np.random.randint(self.num_heads)

    def get_action(self, action_values):
        return EGreedy.get_action(self, action_values[self.selected_head])

    def get_control_param(self):
        return self.selected_head
