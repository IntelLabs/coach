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
import tensorflow as tf


class Bayesian(ExplorationPolicy):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        ExplorationPolicy.__init__(self, tuning_parameters)
        self.keep_probability = tuning_parameters.exploration.initial_keep_probability
        self.final_keep_probability = tuning_parameters.exploration.final_keep_probability
        self.keep_probability_decay_delta = (
                                            tuning_parameters.exploration.initial_keep_probability - tuning_parameters.exploration.final_keep_probability) \
                                            / float(tuning_parameters.exploration.keep_probability_decay_steps)
        self.action_space_size = tuning_parameters.env.action_space_size
        self.network = tuning_parameters.network
        self.epsilon = 0

    def decay_keep_probability(self):
        if (self.keep_probability > self.final_keep_probability and self.keep_probability_decay_delta > 0) \
                or (self.keep_probability < self.final_keep_probability and self.keep_probability_decay_delta < 0):
            self.keep_probability -= self.keep_probability_decay_delta

    def get_action(self, action_values):
        if self.phase == RunPhase.TRAIN:
            self.decay_keep_probability()
        # dropout = self.network.get_layer('variable_dropout_1')
        # with tf.Session() as sess:
        #     print(dropout.rate.eval())
        # set_value(dropout.rate, 1-self.keep_probability)

        print(self.keep_probability)
        self.network.curr_keep_prob = self.keep_probability

        return np.argmax(action_values)

    def get_control_param(self):
        return self.keep_probability
