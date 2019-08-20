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

"""
Module containing utility functions
"""


import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers import Activation
from typing import List, Dict, Any
import numpy as np
import inspect



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

    # Dan change to keras or tf 2 (no need for this)
    # activation_functions = {
    #     'relu': Activation('relu'),
    #     'tanh': Activation('tanh'),
    #     'sigmoid': Activation('sigmoid'),
    #     'elu': Activation('elu'),
    #     'selu': Activation('selu'),
    #     'leaky_relu': Activation('leaky_relu'),
    #     'none': None
    # }
    assert activation_function_string in activation_functions.keys(), \
        "Activation function must be one of the following {}. instead it was: {}" \
            .format(activation_functions.keys(), activation_function_string)
    #return activation_functions[activation_function_string]
    return keras.activations.get(activation_function_string)


def squeeze_tensor(tensor):
    if tensor.shape[0] == 1:
        return tensor[0]
    else:
        return tensor

def get_loss_agent_inputs(inputs: Dict[str, np.ndarray], head_type_idx: int, loss: Any) -> List[np.ndarray]:
    """
    Collects all inputs with prefix 'output_<head_idx>_' and matches them against agent_inputs in loss input schema.
    :param inputs: list of all agent inputs
    :param head_type_idx: head-type index of the corresponding head
    :param loss: corresponding loss
    :return: list of agent inputs for this loss. This list matches the length in loss input schema.
    """
    loss_inputs = list()
    for k in sorted(inputs.keys()):
        if k.startswith('output_{}_'.format(head_type_idx)):
            loss_inputs.append(inputs[k])
    # Enforce that number of inputs for head_type are the same as agent_inputs specified by loss input_schema
    assert len(loss_inputs) == len(loss.input_schema.agent_inputs), "agent_input length doesn't match schema"
    return loss_inputs


def split_outputs_per_head(outputs, heads: list):
    """
    Split outputs into outputs per head
    :param outputs: list of all outputs
    :param heads: list of all heads
    :return: list of outputs for each head
    """
    head_outputs = []
    for h in heads:
        head_outputs.append(list(outputs[:h.num_outputs]))
        outputs = outputs[h.num_outputs:]
    assert len(outputs) == 0
    return head_outputs


def split_targets_per_loss(targets: list, losses: list) -> List[list]:
    """
    Splits targets into targets per loss
    :param targets: list of all targets (typically numpy ndarray)
    :param losses: list of all losses
    :return: list of targets for each loss
    """
    loss_targets = list()
    for l in losses:
        loss_data_len = len(l.input_schema.targets)
        assert len(targets) >= loss_data_len, "Data length doesn't match schema"
        loss_targets.append(targets[:loss_data_len])
        targets = targets[loss_data_len:]
    assert len(targets) == 0
    return loss_targets


# def align_loss_args(
#         head_outputs: List[NDArray],
#         agent_inputs: List[np.ndarray],
#         targets: List[np.ndarray],
#         loss: Any) -> List[np.ndarray]:
#     """
#     Creates a list of arguments from head_outputs, agent_inputs, and targets aligned with parameters of
#     loss.loss_forward() based on their name in loss input_schema
#     :param head_outputs: list of all head_outputs for this loss
#     :param agent_inputs: list of all agent_inputs for this loss
#     :param targets: list of all targets for this loss
#     :param loss: corresponding loss
#     :return: list of arguments in correct order to be passed to loss
#     """
#     arg_list = list()
#     schema = loss.input_schema
#     assert len(schema.head_outputs) == len(head_outputs)
#     assert len(schema.agent_inputs) == len(agent_inputs)
#     assert len(schema.targets) == len(targets)
#
#     prev_found = True
#     for arg_name in inspect.getfullargspec(loss.loss_forward).args[2:]:  # First two args are self and F
#         found = False
#         for schema_list, data in [(schema.head_outputs, head_outputs),
#                                   (schema.agent_inputs, agent_inputs),
#                                   (schema.targets, targets)]:
#             try:
#                 arg_list.append(data[schema_list.index(arg_name)])
#                 found = True
#                 break
#             except ValueError:
#                 continue
#         assert not found or prev_found, "missing arguments detected!"
#         prev_found = found
#     return arg_list