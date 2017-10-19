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


class ThompsonSampling(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.action_space_size = tuning_parameters.env.action_space_size

    def get_action(self, action_values):
        q_values, values_uncertainty = action_values
        sampled_q_values = np.random.normal(q_values, abs(values_uncertainty))
        return np.argmax(sampled_q_values)

    def get_control_param(self):
        return 0
