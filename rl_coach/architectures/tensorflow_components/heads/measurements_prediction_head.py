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

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.architecture import Dense

from rl_coach.architectures.tensorflow_components.heads.head import Head, HeadParameters
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import Measurements
from rl_coach.spaces import SpacesDefinition


class MeasurementsPredictionHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='measurements_prediction_head_params',
                 dense_layer=Dense):
        super().__init__(parameterized_class=MeasurementsPredictionHead,
                         activation_function=activation_function, name=name, dense_layer=dense_layer)


class MeasurementsPredictionHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'future_measurements_head'
        self.num_actions = len(self.spaces.action.actions)
        self.num_measurements = self.spaces.state['measurements'].shape[0]
        self.num_prediction_steps = agent_parameters.algorithm.num_predicted_steps_ahead
        self.multi_step_measurements_size = self.num_measurements * self.num_prediction_steps
        self.return_type = Measurements

    def _build_module(self, input_layer):
        # This is almost exactly the same as Dueling Network but we predict the future measurements for each action
        # actions expectation tower (expectation stream) - E
        with tf.variable_scope("expectation_stream"):
            expectation_stream = self.dense_layer(256)(input_layer, activation=self.activation_function, name='fc1')
            expectation_stream = self.dense_layer(self.multi_step_measurements_size)(expectation_stream, name='output')
            expectation_stream = tf.expand_dims(expectation_stream, axis=1)

        # action fine differences tower (action stream) - A
        with tf.variable_scope("action_stream"):
            action_stream = self.dense_layer(256)(input_layer, activation=self.activation_function, name='fc1')
            action_stream = self.dense_layer(self.num_actions * self.multi_step_measurements_size)(action_stream,
                                                                                                   name='output')
            action_stream = tf.reshape(action_stream,
                                       (tf.shape(action_stream)[0], self.num_actions, self.multi_step_measurements_size))
            action_stream = action_stream - tf.reduce_mean(action_stream, reduction_indices=1, keepdims=True)

        # merge to future measurements predictions
        self.output = tf.add(expectation_stream, action_stream, name='output')
        self.target = tf.placeholder(tf.float32, [None, self.num_actions, self.multi_step_measurements_size],
                                     name="targets")
        targets_nonan = tf.where(tf.is_nan(self.target), self.output, self.target)
        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(targets_nonan - self.output), reduction_indices=0))
        tf.losses.add_loss(self.loss_weight[0] * self.loss)
