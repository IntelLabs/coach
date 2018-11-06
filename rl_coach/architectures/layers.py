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
Module implementing base classes for common network layers used by preset schemes
"""


class Conv2d(object):
    """
    Base class for framework specfic Conv2d layer
    """
    def __init__(self, num_filters: int, kernel_size: int, strides: int):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides

    def __str__(self):
        return "Convolution (num filters = {}, kernel size = {}, stride = {})"\
            .format(self.num_filters, self.kernel_size, self.strides)


class BatchnormActivationDropout(object):
    """
    Base class for framework specific batchnorm->activation->dropout layer group
    """
    def __init__(self, batchnorm: bool=False, activation_function: str=None, dropout_rate: float=0):
        self.batchnorm = batchnorm
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

    def __str__(self):
        result = []
        if self.batchnorm:
            result += ["Batch Normalization"]
        if self.activation_function:
            result += ["Activation (type = {})".format(self.activation_function)]
        if self.dropout_rate > 0:
            result += ["Dropout (rate = {})".format(self.dropout_rate)]
        return "\n".join(result)


class Dense(object):
    """
    Base class for framework specific Dense layer
    """
    def __init__(self, units: int):
        self.units = units

    def __str__(self):
        return "Dense (num outputs = {})".format(self.units)


class NoisyNetDense(object):
    """
    Base class for framework specific factorized Noisy Net layer

    https://arxiv.org/abs/1706.10295.
    """

    def __init__(self, units: int):
        self.units = units
        self.sigma0 = 0.5

    def __str__(self):
        return "Noisy Dense (num outputs = {})".format(self.units)
