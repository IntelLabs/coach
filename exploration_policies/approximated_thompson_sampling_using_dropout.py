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

from exploration_policies import exploration_policy


class ApproximatedThompsonSamplingUsingDropout(exploration_policy.ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        exploration_policy.ExplorationPolicy.__init__(self, tuning_parameters)
        self.dropout_discard_probability = tuning_parameters.exploration.dropout_discard_probability
        self.network = tuning_parameters.network
        self.assign_op = self.network.dropout_discard_probability.assign(self.dropout_discard_probability)
        self.network.sess.run(self.assign_op)
        pass

    def decay_dropout(self):
        pass

    def get_action(self, action_values):
        return np.argmax(action_values)

    def get_control_param(self):
        return self.dropout_discard_probability
