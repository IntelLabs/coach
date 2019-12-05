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


from tensorflow import keras
from typing import Dict, List, Tuple
from tensorflow import Tensor
import numpy as np
import inspect

LOSS_OUT_TYPE_LOSS = 'loss'
LOSS_OUT_TYPE_REGULARIZATION = 'regularization'


class LossInputSchema(object):
    """
    Helper class to contain schema for loss input
    """
    def __init__(self, head_outputs: List[str], agent_inputs: List[str], targets: List[str]):
        """
        :param head_outputs: list of argument names in call that are the outputs of the head.
            The order and number MUST MATCH the output from the head.
        :param agent_inputs: list of argument names that are inputs from the agent.
            The order and number MUST MATCH `output_<head_type_idx>_<order>` for this head.
        :param targets: list of argument names that are targets for the loss.
            The order and number MUST MATCH targets passed from the agent.
        """
        self._head_outputs = head_outputs
        self._agent_inputs = agent_inputs
        self._targets = targets

    @property
    def head_outputs(self):
        return self._head_outputs

    @property
    def agent_inputs(self):
        return self._agent_inputs

    @property
    def targets(self):
        return self._targets


class HeadLoss(keras.layers.Layer):
    """
    ABC for loss functions of each agent. Child class must implement input_schema() and loss_forward()
    """

    def __init__(self, *args, **kwargs):
        super(HeadLoss, self).__init__(*args, **kwargs)
        self._output_schema = None  # type: List[str]

    @property
    def input_schema(self) -> LossInputSchema:
        """
        :return: schema for input of loss forward. Read docstring for LossInputSchema for details.
        """
        raise NotImplementedError

    @property
    def output_schema(self) -> List[str]:
        """
        :return: schema for output. Must contain 'loss' and 'regularization' keys at least once.
            The order and total number must match that of returned values from the loss. 'loss' and 'regularization'
            are special keys. Any other string is treated as auxiliary outputs and must include match auxiliary
            fetch names returned by the head.
        """
        return self._output_schema

    def _loss_output(self, outputs: List[Tuple[Tensor, str]]):
        """
        Must be called on the output from call ().
        Saves the returned output as the schema and returns output values in a list
        :return: list of output values
        """
        output_schema = [o[1] for o in outputs]
        assert self._output_schema is None or self._output_schema == output_schema
        self._output_schema = output_schema
        return tuple(o[0] for o in outputs)

    def call(self, head_output, agent_input, target):
        loss_args = self.extract_loss_args(head_output, agent_input, target)
        loss_output = self._loss_output(self.loss_forward(*loss_args))
        return self.loss_output_dict(loss_output)

    def loss_forward(self, *args, **kwargs):
        raise NotImplementedError

    def extract_loss_args(self,
                          head_outputs: List[Tensor],
                          agent_inputs: List[np.ndarray],
                          targets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Creates a list of arguments from head_outputs, agent_inputs, and targets aligned with parameters of
        loss.loss_forward() based on their name in loss input_schema
        :param head_outputs: list of all head_outputs for this loss
        :param agent_inputs: list of all agent_inputs for this loss
        :param targets: list of all targets for this loss
        :return: list of arguments in correct order to be passed to loss
        """
        arg_list = list()
        schema = self.input_schema
        assert len(schema.head_outputs) == len(head_outputs)
        assert len(schema.agent_inputs) == len(agent_inputs)
        assert len(schema.targets) == len(targets)

        for arg_name in inspect.getfullargspec(self.loss_forward).args[1:]:  # First argument is self
            for schema_list, data in [(schema.head_outputs, head_outputs),
                                      (schema.agent_inputs, agent_inputs),
                                      (schema.targets, targets)]:
                try:
                    # Index of loss function argument in the corresponding part of the loss input schema
                    schema_index = schema_list.index(arg_name)
                    arg_list.append(data[schema_index])
                    break
                except ValueError:
                    continue
        return arg_list

    def loss_output_dict(self, output: List) -> Dict[str, List]:
        """
        Creates a dictionary for loss output based on the output schema. If two output values have the same
        type string in the schema they are concatenated in the same dictionary item.
        :param output: list of output values
        :param schema: list of type-strings for output values
        :return: dictionary of keyword to list of NDArrays
        """
        schema = self.output_schema
        assert len(output) == len(schema)
        output_dict = dict()
        for name, val in zip(schema, output):
            if name in output_dict:
                output_dict[name].append(val)
            else:
                output_dict[name] = [val]
        return output_dict
