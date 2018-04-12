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


class Architecture(object):
    def __init__(self, tuning_parameters, name=""):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        :param name: The name of the network
        :param name: string
        """
        self.batch_size = tuning_parameters.batch_size
        self.input_depth = tuning_parameters.env.observation_stack_size
        self.input_height = tuning_parameters.env.desired_observation_height
        self.input_width = tuning_parameters.env.desired_observation_width
        self.num_actions = tuning_parameters.env.action_space_size
        self.measurements_size = tuning_parameters.env.measurements_size \
            if tuning_parameters.env.measurements_size else 0
        self.learning_rate = tuning_parameters.learning_rate
        self.optimizer = None
        self.name = name
        self.tp = tuning_parameters

    def get_model(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running parameters
        :type tuning_parameters: Preset
        :return: A model
        """
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
