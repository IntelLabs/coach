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

from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class Architecture(object):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str= ""):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces (observation, action, etc.) definition of the agent
        :param name: the name of the network
        """
        # spaces
        self.spaces = spaces

        self.name = name
        self.network_wrapper_name = self.name.split('/')[0]  # the name can be main/online and the network_wrapper_name will be main
        self.full_name = "{}/{}".format(agent_parameters.full_name_id, name)
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]
        self.batch_size = self.network_parameters.batch_size
        self.learning_rate = self.network_parameters.learning_rate
        self.optimizer = None
        self.ap = agent_parameters

    def get_model(self):
        pass

    def predict(self, inputs):
        pass

    def train_on_batch(self, inputs, targets):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights, rate=1.0):
        pass

    def reset_accumulated_gradients(self):
        pass

    def accumulate_gradients(self, inputs, targets):
        pass

    def apply_and_reset_gradients(self, gradients):
        pass

    def apply_gradients(self, gradients):
        pass

    def get_variable_value(self, variable):
        pass

    def set_variable_value(self, assign_op, value, placeholder=None):
        pass
