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


import tensorflow as tf
import numpy as np
from tensorflow import keras
from typing import List
from typing import Dict, Union, Any
from tensorflow import Tensor


"""
Module containing utility functions
"""
LOSS_OUT_TYPE_LOSS = 'loss'
LOSS_OUT_TYPE_REGULARIZATION = 'regularization'


def get_activation_function(activation_function_string: str):
    """
    Map the activation function from a string to the tensorflow framework equivalent
    :param activation_function_string: the type of the activation function
    :return: the tensorflow activation function
    """

    activation_functions = {
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu,
        'leaky_relu': tf.nn.leaky_relu,
        'none': None
    }

    assert activation_function_string in activation_functions.keys(), \
        "Activation function must be one of the following {}. instead it was: {}" \
            .format(activation_functions.keys(), activation_function_string)
    return keras.activations.get(activation_function_string)


def squeeze_tensor(tensor):
    if tensor.shape[0] == 1:
        return tensor[0]
    else:
        return tensor


def to_list(data: Union[tuple, list, Any]):
    """
    If input is tuple, it is converted to list. If it's list, it is returned untouched. Otherwise
    returns a single-element list of the data.
    :return: list-ified data
    """
    if isinstance(data, list):
        pass
    elif isinstance(data, tuple):
        data = list(data)
    else:
        data = [data]
    return data


def loss_output_dict(output: List[Tensor], schema: List[str]) -> Dict[str, List[Tensor]]:
    """
    Creates a dictionary for loss output based on the output schema. If two output values have the same
    type string in the schema they are concatenated in the same dicrionary item.
    :param output: list of output values
    :param schema: list of type-strings for output values
    :return: dictionary of keyword to list of NDArrays
    """
    assert len(output) == len(schema)
    output_dict = dict()
    for name, val in zip(schema, output):
        if name in output_dict:
            output_dict[name].append(val)
        else:
            output_dict[name] = [val]
    return output_dict


def extract_loss_inputs(head_index, inputs, targets: List[np.ndarray]) -> List[np.ndarray]:
    """
    Creates a list of arguments from model_outputs and non_trainable_args aligned with parameters of
    loss.loss_forward() based on their name in loss input_schema.
    :param model_outputs: list of all trainable model_outputs for this loss
    :param non_trainable_args: list of all non trainable args for this loss
    :return: list of arguments in correct order to be passed to loss
    """
    arg_list = list()
    non_trainable_args = filter(lambda elem: elem[0].startswith('output_{}_'.format(head_index)), inputs.items())
    non_trainable_args = dict(non_trainable_args)
    non_trainable = []
    for key in sorted(non_trainable_args.keys()):
        non_trainable.append(non_trainable_args[key])

    if non_trainable:
        non_trainable_args = non_trainable + [targets[head_index]]
    else:
        non_trainable_args = [targets[head_index]]

    return non_trainable_args


def extract_fetches(loss_outputs: List[Tensor], head_idx, additional_fetches):
    """
    Creates a dictionary for loss output based on the output schema. If two output values have the same
    type string in the schema they are concatenated in the same dicrionary item.
    :param output: list of output values
    :param schema: list of type-strings for output values
    :return: dictionary of keyword to list of NDArrays
    """
    additional_fetches = [(k, None) for k in additional_fetches]

    for i, fetch in enumerate(additional_fetches):
        head_type_idx, fetch_name = fetch[0]  # fetch key is a tuple of (head_type_index, fetch_name)
        if head_idx == head_type_idx:
            assert fetch[1] is None  # sanity check that fetch is None
            additional_fetches[i] = (fetch[0], loss_outputs[fetch_name])

    # result of of additional fetches
    fetched_tensors = [fetch[1] for fetch in additional_fetches]
    return fetched_tensors
