#
# Copyright (c) 2019 Intel Corporation
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
    def __init__(self, model_outputs: List[str], non_trainable_args: List[str]):
        """
        :param model_outputs: list of argument names in call that are the outputs of the head.
            The order and number MUST MATCH the output from the head.
        :param non_trainable_args: list of argument names that are inputs from the agent.
            The order and number MUST MATCH `output_<head_type_idx>_<order>` for this head.
        :param targets: list of argument names that are targets for the loss.
            The order and number MUST MATCH targets passed from the agent.
        """
        self._model_outputs = model_outputs
        self._non_trainable_args = non_trainable_args

    @property
    def model_outputs(self):
        return self._model_outputs

    @property
    def non_trainable_args(self):
        return self._non_trainable_args


class HeadLoss(keras.layers.Layer):
    """
    ABC for loss functions of each agent. Child class must implement input_schema() and loss_forward()
    """

    def __init__(self, *args, **kwargs):
        super(HeadLoss, self).__init__(*args, **kwargs)

    @property
    def input_schema(self) -> LossInputSchema:
        """
        :return: schema for input of loss forward. Read docstring for LossInputSchema for details.
        """
        raise NotImplementedError

    # def call(self, model_outputs, non_trainable_args):
    #     loss_args = self.extract_loss_args(model_outputs, non_trainable_args)
    #     return self.loss_forward(*loss_args)

    def call(self, model_outputs, targets, inputs):
        non_trainable_args = self.extract_loss_args(inputs, targets)

        loss_args = self.align_loss_args(model_outputs, non_trainable_args)

        # loss_value, regularisation, additional_info = self.extract_loss_outputs(self.loss_forward(*loss_args), additional_fetches)
        #
        # return loss_value, regularisation, additional_info

        return self.loss_forward(*loss_args)


    def loss_forward(self, *args, **kwargs):
        raise NotImplementedError



    def extract_loss_args(self, inputs, targets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Creates a list of arguments from model_outputs and non_trainable_args aligned with parameters of
        loss.loss_forward() based on their name in loss input_schema.
        :param model_outputs: list of all trainable model_outputs for this loss
        :param non_trainable_args: list of all non trainable args for this loss
        :return: list of arguments in correct order to be passed to loss
        """
        arg_list = list()
        non_trainable_args = filter(lambda elem: elem[0].startswith('output_{}_'.format(self.head_idx)), inputs.items())
        non_trainable_args = dict(non_trainable_args)
        non_trainable = []
        for key in sorted(non_trainable_args.keys()):
            non_trainable.append(non_trainable_args[key])

        if non_trainable:
            non_trainable_args = non_trainable + [targets[self.head_idx]]
        else:
            non_trainable_args = [targets[self.head_idx]]

        return non_trainable_args


    def extract_loss_outputs(self, loss_outputs, additional_fetches) -> List[np.ndarray]:
        """
        Creates a list of arguments from model_outputs and non_trainable_args aligned with parameters of
        loss.loss_forward() based on their name in loss input_schema.
        :param model_outputs: list of all trainable model_outputs for this loss
        :param non_trainable_args: list of all non trainable args for this loss
        :return: list of arguments in correct order to be passed to loss
        """
        arg_list = list()
        #additional_fetches = []
        loss_value = None
        regularisation = None

        additional_fetches = [(k, None) for k in additional_fetches]

        if LOSS_OUT_TYPE_LOSS in loss_outputs:
            loss_value = loss_outputs[LOSS_OUT_TYPE_LOSS]
        if LOSS_OUT_TYPE_REGULARIZATION in loss_outputs:
            regularisation = loss_outputs[LOSS_OUT_TYPE_REGULARIZATION]

        for i, fetch in enumerate(additional_fetches):
            head_type_idx, fetch_name = fetch[0]  # fetch key is a tuple of (head_type_index, fetch_name)
            if self.head_idx == head_type_idx:
                assert fetch[1] is None  # sanity check that fetch is None
                additional_fetches[i] = (fetch[0], loss_outputs[fetch_name])

        return loss_value, regularisation, additional_fetches


    def align_loss_args(self,
                          model_outputs: List[Tensor],
                          non_trainable_args: List[np.ndarray]) -> List[np.ndarray]:
        """
        Creates a list of arguments from model_outputs and non_trainable_args aligned with parameters of
        loss.loss_forward() based on their name in loss input_schema.
        :param model_outputs: list of all trainable model_outputs for this loss
        :param non_trainable_args: list of all non trainable args for this loss
        :return: list of arguments in correct order to be passed to loss
        """
        arg_list = list()
        schema = self.input_schema
        assert len(schema.model_outputs) == len(model_outputs)
        assert len(schema.non_trainable_args) == len(non_trainable_args)

        for arg_name in inspect.getfullargspec(self.loss_forward).args[1:]:  # First argument is self
            for schema_list, data in [(schema.model_outputs, model_outputs),
                                      (schema.non_trainable_args, non_trainable_args)]:
                try:
                    # Index of loss function argument in the corresponding part of the loss input schema
                    schema_index = schema_list.index(arg_name)
                    arg_list.append(data[schema_index])
                    break
                except ValueError:
                    continue
        return arg_list


    # def extract_loss_args(self,
    #                       model_outputs: List[Tensor],
    #                       non_trainable_args: List[np.ndarray]) -> List[np.ndarray]:
    #     """
    #     Creates a list of arguments from model_outputs and non_trainable_args aligned with parameters of
    #     loss.loss_forward() based on their name in loss input_schema.
    #     :param model_outputs: list of all trainable model_outputs for this loss
    #     :param non_trainable_args: list of all non trainable args for this loss
    #     :return: list of arguments in correct order to be passed to loss
    #     """
    #     arg_list = list()
    #     schema = self.input_schema
    #     assert len(schema.model_outputs) == len(model_outputs)
    #     assert len(schema.non_trainable_args) == len(non_trainable_args)
    #
    #     for arg_name in inspect.getfullargspec(self.loss_forward).args[1:]:  # First argument is self
    #         for schema_list, data in [(schema.model_outputs, model_outputs),
    #                                   (schema.non_trainable_args, non_trainable_args)]:
    #             try:
    #                 # Index of loss function argument in the corresponding part of the loss input schema
    #                 schema_index = schema_list.index(arg_name)
    #                 arg_list.append(data[schema_index])
    #                 break
    #             except ValueError:
    #                 continue
    #     return arg_list
