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

import sys
import copy
from ngraph.frontends.neon import *
import ngraph as ng
from architectures.architecture import *
import numpy as np
from utils import *


class NeonArchitecture(Architecture):
    def __init__(self, tuning_parameters, name="", global_network=None, network_is_local=True):
        Architecture.__init__(self, tuning_parameters, name)
        assert tuning_parameters.agent.neon_support, 'Neon is not supported for this agent'
        self.clip_error = tuning_parameters.clip_gradients
        self.total_loss = None
        self.epoch = 0
        self.inputs = []
        self.outputs = []
        self.targets = []
        self.losses = []

        self.transformer = tuning_parameters.sess
        self.network = self.get_model(tuning_parameters)
        self.accumulated_gradients = []

        # training and inference ops
        train_output = ng.sequential([
            self.optimizer(self.total_loss),
            self.total_loss
        ])
        placeholders = self.inputs + self.targets
        self.train_op = self.transformer.add_computation(
            ng.computation(
                train_output, *placeholders
            )
        )
        self.predict_op = self.transformer.add_computation(
            ng.computation(
                self.outputs, self.inputs[0]
            )
        )

        # update weights from array op
        self.weights = [ng.placeholder(w.axes) for w in self.total_loss.variables()]
        self.set_weights_ops = []
        for target_variable, variable in zip(self.total_loss.variables(), self.weights):
            self.set_weights_ops.append(self.transformer.add_computation(
                ng.computation(
                    ng.assign(target_variable, variable), variable
                )
            ))

        # get weights op
        self.get_variables = self.transformer.add_computation(
            ng.computation(
                self.total_loss.variables()
            )
        )

    def predict(self, inputs):
        batch_size = inputs.shape[0]

        # move batch axis to the end
        inputs = inputs.swapaxes(0, -1)
        prediction = self.predict_op(inputs)  # TODO: problem with multiple inputs

        if type(prediction) != tuple:
            prediction = (prediction)

        # process all the outputs from the network
        output = []
        for p in prediction:
            output.append(p.transpose()[:batch_size].copy())

        # if there is only one output then we don't need a list
        if len(output) == 1:
            output = output[0]
        return output

    def train_on_batch(self, inputs, targets):
        loss = self.accumulate_gradients(inputs, targets)
        self.apply_and_reset_gradients(self.accumulated_gradients)
        return loss

    def get_weights(self):
        return self.get_variables()

    def set_weights(self, weights, rate=1.0):
        if rate != 1:
            current_weights = self.get_weights()
            updated_weights = [(1 - rate) * t + rate * o for t, o in zip(current_weights, weights)]
        else:
            updated_weights = weights
        for update_function, variable in zip(self.set_weights_ops, updated_weights):
            update_function(variable)

    def accumulate_gradients(self, inputs, targets):
        # Neon doesn't currently allow separating the grads calculation and grad apply operations
        # so this feature is not currently available. instead we do a full training iteration
        inputs = force_list(inputs)
        targets = force_list(targets)

        for idx, input in enumerate(inputs):
            inputs[idx] = input.swapaxes(0, -1)

        for idx, target in enumerate(targets):
            targets[idx] = np.rollaxis(target, 0, len(target.shape))

        all_inputs = inputs + targets

        loss = np.mean(self.train_op(*all_inputs))

        return [loss]
