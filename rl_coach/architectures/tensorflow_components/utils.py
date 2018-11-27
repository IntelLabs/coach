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
    return activation_functions[activation_function_string]
