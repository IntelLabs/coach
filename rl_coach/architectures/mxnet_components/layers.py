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
Module implementing basic layers in mxnet
"""

from types import FunctionType

from mxnet.gluon import nn

from rl_coach.architectures import layers
from rl_coach.architectures.mxnet_components import utils


# define global dictionary for storing layer type to layer implementation mapping
mx_layer_dict = dict()


def reg_to_mx(layer_type) -> FunctionType:
    """ function decorator that registers layer implementation
    :return: decorated function
    """
    def reg_impl_decorator(func):
        assert layer_type not in mx_layer_dict
        mx_layer_dict[layer_type] = func
        return func
    return reg_impl_decorator


def convert_layer(layer):
    """
    If layer is callable, return layer, otherwise convert to MX type
    :param layer: layer to be converted
    :return: converted layer if not callable, otherwise layer itself
    """
    if callable(layer):
        return layer
    return mx_layer_dict[type(layer)](layer)


class Conv2d(layers.Conv2d):
    def __init__(self, num_filters: int, kernel_size: int, strides: int):
        super(Conv2d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides)

    def __call__(self) -> nn.Conv2D:
        """
        returns a conv2d block
        :return: conv2d block
        """
        return nn.Conv2D(channels=self.num_filters, kernel_size=self.kernel_size, strides=self.strides)

    @staticmethod
    @reg_to_mx(layers.Conv2d)
    def to_mx(base: layers.Conv2d):
        return Conv2d(num_filters=base.num_filters, kernel_size=base.kernel_size, strides=base.strides)


class BatchnormActivationDropout(layers.BatchnormActivationDropout):
    def __init__(self, batchnorm: bool=False, activation_function=None, dropout_rate: float=0):
        super(BatchnormActivationDropout, self).__init__(
            batchnorm=batchnorm, activation_function=activation_function, dropout_rate=dropout_rate)

    def __call__(self):
        """
        returns a list of mxnet batchnorm, activation and dropout layers
        :return: batchnorm, activation and dropout layers
        """
        block = nn.HybridSequential()
        if self.batchnorm:
            block.add(nn.BatchNorm())
        if self.activation_function:
            block.add(nn.Activation(activation=utils.get_mxnet_activation_name(self.activation_function)))
        if self.dropout_rate:
            block.add(nn.Dropout(self.dropout_rate))
        return block

    @staticmethod
    @reg_to_mx(layers.BatchnormActivationDropout)
    def to_mx(base: layers.BatchnormActivationDropout):
        return BatchnormActivationDropout(
            batchnorm=base.batchnorm,
            activation_function=base.activation_function,
            dropout_rate=base.dropout_rate)


class Dense(layers.Dense):
    def __init__(self, units: int):
        super(Dense, self).__init__(units=units)

    def __call__(self):
        """
        returns a mxnet dense layer
        :return: dense layer
        """
        # Set flatten to False for consistent behavior with tf.layers.dense
        return nn.Dense(self.units, flatten=False)

    @staticmethod
    @reg_to_mx(layers.Dense)
    def to_mx(base: layers.Dense):
        return Dense(units=base.units)
