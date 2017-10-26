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

import ngraph as ng
from ngraph.util.names import name_scope
import ngraph.frontends.neon as neon
import numpy as np
from utils import force_list
from architectures.neon_components.losses import *


class Head(object):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        self.head_idx = head_idx
        self.name = "head"
        self.output = []
        self.loss = []
        self.loss_type = []
        self.regularizations = []
        self.loss_weight = force_list(loss_weight)
        self.weights_init = neon.GlorotInit()
        self.biases_init = neon.ConstantInit()
        self.target = []
        self.input = []
        self.is_local = is_local
        self.batch_size = tuning_parameters.batch_size

    def __call__(self, input_layer):
        """
        Wrapper for building the module graph including scoping and loss creation
        :param input_layer: the input to the graph
        :return: the output of the last layer and the target placeholder
        """
        with name_scope(self.get_name()):
            self._build_module(input_layer)

            self.output = force_list(self.output)
            self.target = force_list(self.target)
            self.input = force_list(self.input)
            self.loss_type = force_list(self.loss_type)
            self.loss = force_list(self.loss)
            self.regularizations = force_list(self.regularizations)
            if self.is_local:
               self.set_loss()

        if self.is_local:
            return self.output, self.target, self.input
        else:
            return self.output, self.input

    def _build_module(self, input_layer):
        """
        Builds the graph of the module
        :param input_layer: the input to the graph
        :return: None
        """
        pass

    def get_name(self):
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return '{}_{}'.format(self.name, self.head_idx)

    def set_loss(self):
        """
        Creates a target placeholder and loss function for each loss_type and regularization
        :param loss_type: a tensorflow loss function
        :param scope: the name scope to include the tensors in
        :return: None
        """
        # add losses and target placeholder
        for idx in range(len(self.loss_type)):
            # output_axis = ng.make_axis(self.num_actions, name='q_values')
            batch_axis_full = ng.make_axis(self.batch_size, name='N')
            target = ng.placeholder(ng.make_axes([self.output[0].axes[0], batch_axis_full]))
            self.target.append(target)
            loss = self.loss_type[idx](self.target[-1], self.output[idx],
                                       weights=self.loss_weight[idx], scope=self.get_name())
            self.loss.append(loss)

        # add regularizations
        for regularization in self.regularizations:
            self.loss.append(regularization)


class QHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'q_values_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            raise Exception("huber loss is not supported in neon")
        else:
            self.loss_type = mean_squared_error

    def _build_module(self, input_layer):
        # Standard Q Network
        self.output = neon.Sequential([
                neon.Affine(nout=self.num_actions,
                            weight_init=self.weights_init, bias_init=self.biases_init)
            ])(input_layer)


class DuelingQHead(QHead):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        QHead.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)

    def _build_module(self, input_layer):
        # Dueling Network
        # state value tower - V
        output_axis = ng.make_axis(self.num_actions, name='q_values')

        state_value = neon.Sequential([
            neon.Affine(nout=256, activation=neon.Rectlin(),
                        weight_init=self.weights_init, bias_init=self.biases_init),
            neon.Affine(nout=1,
                        weight_init=self.weights_init, bias_init=self.biases_init)
        ])(input_layer)

        # action advantage tower - A
        action_advantage_unnormalized = neon.Sequential([
            neon.Affine(nout=256, activation=neon.Rectlin(),
                        weight_init=self.weights_init, bias_init=self.biases_init),
            neon.Affine(axes=output_axis,
                        weight_init=self.weights_init, bias_init=self.biases_init)
        ])(input_layer)
        action_advantage = action_advantage_unnormalized - ng.mean(action_advantage_unnormalized)

        repeated_state_value = ng.expand_dims(ng.slice_along_axis(state_value, state_value.axes[0], 0), output_axis, 0)

        # merge to state-action value function Q
        self.output = repeated_state_value + action_advantage


class MeasurementsPredictionHead(Head):
    def __init__(self, tuning_parameters, head_idx=0, loss_weight=1., is_local=True):
        Head.__init__(self, tuning_parameters, head_idx, loss_weight, is_local)
        self.name = 'future_measurements_head'
        self.num_actions = tuning_parameters.env_instance.action_space_size
        self.num_measurements = tuning_parameters.env.measurements_size[0] \
            if tuning_parameters.env.measurements_size else 0
        self.num_prediction_steps = tuning_parameters.agent.num_predicted_steps_ahead
        self.multi_step_measurements_size = self.num_measurements * self.num_prediction_steps
        if tuning_parameters.agent.replace_mse_with_huber_loss:
            raise Exception("huber loss is not supported in neon")
        else:
            self.loss_type = mean_squared_error

    def _build_module(self, input_layer):
        # This is almost exactly the same as Dueling Network but we predict the future measurements for each action

        multistep_measurements_size = self.measurements_size[0] * self.num_predicted_steps_ahead

        # actions expectation tower (expectation stream) - E
        with name_scope("expectation_stream"):
            expectation_stream = neon.Sequential([
                neon.Affine(nout=256, activation=neon.Rectlin(),
                            weight_init=self.weights_init, bias_init=self.biases_init),
                neon.Affine(nout=multistep_measurements_size,
                            weight_init=self.weights_init, bias_init=self.biases_init)
            ])(input_layer)

        # action fine differences tower (action stream) - A
        with name_scope("action_stream"):
            action_stream_unnormalized = neon.Sequential([
                neon.Affine(nout=256, activation=neon.Rectlin(),
                            weight_init=self.weights_init, bias_init=self.biases_init),
                neon.Affine(nout=self.num_actions * multistep_measurements_size,
                            weight_init=self.weights_init, bias_init=self.biases_init),
                neon.Reshape((self.num_actions, multistep_measurements_size))
            ])(input_layer)
            action_stream = action_stream_unnormalized - ng.mean(action_stream_unnormalized)

        repeated_expectation_stream = ng.slice_along_axis(expectation_stream, expectation_stream.axes[0], 0)
        repeated_expectation_stream = ng.expand_dims(repeated_expectation_stream, output_axis, 0)

        # merge to future measurements predictions
        self.output = repeated_expectation_stream + action_stream

