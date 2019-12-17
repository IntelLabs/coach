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
            The order and number MUST MATCH the loss_forward call for this head.

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

    def call(self, model_outputs: List[Tensor], non_trainable_args: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extracts and aligns loss arguments and Passes the cal to loss_forward()
        :param model_outputs: list of all trainable model_outputs for this loss
        :param non_trainable_args: list of all non trainable args for this loss
        :return: list of arguments in containing the loss values regularization values and additional fetches.
        """
        loss_args = self.align_loss_args(model_outputs, non_trainable_args)
        return self.loss_forward(*loss_args)

    def loss_forward(self, *args, **kwargs):
        """
        Needs to be implemented by each child class
        """
        raise NotImplementedError

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

        arg_list.extend(model_outputs)
        arg_list.extend(non_trainable_args)
        return arg_list
