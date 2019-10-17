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
from tensorflow import keras
from typing import List
from typing import Dict, Union, Any
from tensorflow import Tensor


"""
Module containing utility functions
"""



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
        # Remove associated outputs
        outputs = outputs[h.num_outputs:]
    # A check that we don't forget something
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


# def get_loss_agent_inputs(inputs: Dict[str, np.ndarray], head_type_idx: int, loss: Any) -> List[np.ndarray]:
#     """
#     Collects all inputs with prefix 'output_<head_idx>_' and matches them against agent_inputs in loss input schema.
#     :param inputs: list of all agent inputs
#     :param head_type_idx: head-type index of the corresponding head
#     :param loss: corresponding loss
#     :return: list of agent inputs for this loss. This list matches the length in loss input schema.
#     """
#     loss_inputs = list()
#     for k in sorted(inputs.keys()):
#         if k.startswith('output_{}_'.format(head_type_idx)):
#             loss_inputs.append(inputs[k])
#     # Enforce that number of inputs for head_type are the same as agent_inputs specified by loss input_schema
#     assert len(loss_inputs) == len(loss.input_schema.agent_inputs), "agent_input length doesn't match schema"
#     return loss_inputs


