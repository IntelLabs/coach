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


import math
from types import FunctionType
from typing import Any

import tensorflow as tf

from rl_coach.architectures import layers
from rl_coach.architectures.tensorflow_components import utils


def batchnorm_activation_dropout(input_layer, batchnorm, activation_function, dropout_rate, is_training, name):
    layers = [input_layer]

    # batchnorm
    if batchnorm:
        layers.append(
            tf.layers.batch_normalization(layers[-1], name="{}_batchnorm".format(name), training=is_training)
        )

    # activation
    if activation_function:
        if isinstance(activation_function, str):
            activation_function = utils.get_activation_function(activation_function)
        layers.append(
            activation_function(layers[-1], name="{}_activation".format(name))
        )

    # dropout
    if dropout_rate > 0:
        layers.append(
            tf.layers.dropout(layers[-1], dropout_rate, name="{}_dropout".format(name), training=is_training)
        )

    # remove the input layer from the layers list
    del layers[0]

    return layers


# define global dictionary for storing layer type to layer implementation mapping
tf_layer_dict = dict()


def reg_to_tf(layer_type) -> FunctionType:
    """ function decorator that registers layer implementation
    :return: decorated function
    """
    def reg_impl_decorator(func):
        assert layer_type not in tf_layer_dict
        tf_layer_dict[layer_type] = func
        return func
    return reg_impl_decorator


def convert_layer(layer):
    """
    If layer is callable, return layer, otherwise convert to TF type
    :param layer: layer to be converted
    :return: converted layer if not callable, otherwise layer itself
    """
    if callable(layer):
        return layer
    return tf_layer_dict[type(layer)](layer)


class Conv2d(layers.Conv2d):
    def __init__(self, num_filters: int, kernel_size: int, strides: int):
        super(Conv2d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides)

    def __call__(self, input_layer, name: str=None, is_training=None):
        """
        returns a tensorflow conv2d layer
        :param input_layer: previous layer
        :param name: layer name
        :return: conv2d layer
        """
        return tf.layers.conv2d(input_layer, filters=self.num_filters, kernel_size=self.kernel_size,
                                strides=self.strides, data_format='channels_last', name=name)

    @staticmethod
    @reg_to_tf(layers.Conv2d)
    def to_tf(base: layers.Conv2d):
        return Conv2d(
            num_filters=base.num_filters,
            kernel_size=base.kernel_size,
            strides=base.strides)


class BatchnormActivationDropout(layers.BatchnormActivationDropout):
    def __init__(self, batchnorm: bool=False, activation_function=None, dropout_rate: float=0):
        super(BatchnormActivationDropout, self).__init__(
            batchnorm=batchnorm, activation_function=activation_function, dropout_rate=dropout_rate)

    def __call__(self, input_layer, name: str=None, is_training=None):
        """
        returns a list of tensorflow batchnorm, activation and dropout layers
        :param input_layer: previous layer
        :param name: layer name
        :return: batchnorm, activation and dropout layers
        """
        return batchnorm_activation_dropout(input_layer, batchnorm=self.batchnorm,
                                            activation_function=self.activation_function,
                                            dropout_rate=self.dropout_rate,
                                            is_training=is_training, name=name)

    @staticmethod
    @reg_to_tf(layers.BatchnormActivationDropout)
    def to_tf(base: layers.BatchnormActivationDropout):
        return BatchnormActivationDropout(
            batchnorm=base.batchnorm,
            activation_function=base.activation_function,
            dropout_rate=base.dropout_rate)


class Dense(layers.Dense):
    def __init__(self, units: int):
        super(Dense, self).__init__(units=units)

    def __call__(self, input_layer, name: str=None, kernel_initializer=None, activation=None, is_training=None):
        """
        returns a tensorflow dense layer
        :param input_layer: previous layer
        :param name: layer name
        :return: dense layer
        """
        return tf.layers.dense(input_layer, self.units, name=name, kernel_initializer=kernel_initializer,
                               activation=activation)

    @staticmethod
    @reg_to_tf(layers.Dense)
    def to_tf(base: layers.Dense):
        return Dense(units=base.units)


class NoisyNetDense(layers.NoisyNetDense):
    """
    A factorized Noisy Net layer

    https://arxiv.org/abs/1706.10295.
    """

    def __init__(self, units: int):
        super(NoisyNetDense, self).__init__(units=units)

    def __call__(self, input_layer, name: str, kernel_initializer=None, activation=None, is_training=None):
        """
        returns a NoisyNet dense layer
        :param input_layer: previous layer
        :param name: layer name
        :param kernel_initializer: initializer for kernels. Default is to use Gaussian noise that preserves stddev.
        :param activation: the activation function
        :return: dense layer
        """
        #TODO: noise sampling should be externally controlled. DQN is fine with sampling noise for every
        #      forward (either act or train, both for online and target networks).
        #      A3C, on the other hand, should sample noise only when policy changes (i.e. after every t_max steps)

        def _f(values):
            return tf.sqrt(tf.abs(values)) * tf.sign(values)

        def _factorized_noise(inputs, outputs):
            # TODO: use factorized noise only for compute intensive algos (e.g. DQN).
            #      lighter algos (e.g. DQN) should not use it
            noise1 = _f(tf.random_normal((inputs, 1)))
            noise2 = _f(tf.random_normal((1, outputs)))
            return tf.matmul(noise1, noise2)

        num_inputs = input_layer.get_shape()[-1].value
        num_outputs = self.units

        stddev = 1 / math.sqrt(num_inputs)
        activation = activation if activation is not None else (lambda x: x)

        if kernel_initializer is None:
            kernel_mean_initializer = tf.random_uniform_initializer(-stddev, stddev)
            kernel_stddev_initializer = tf.random_uniform_initializer(-stddev * self.sigma0, stddev * self.sigma0)
        else:
            kernel_mean_initializer = kernel_stddev_initializer = kernel_initializer
        with tf.variable_scope(None, default_name=name):
            weight_mean = tf.get_variable('weight_mean', shape=(num_inputs, num_outputs),
                                          initializer=kernel_mean_initializer)
            bias_mean = tf.get_variable('bias_mean', shape=(num_outputs,), initializer=tf.zeros_initializer())

            weight_stddev = tf.get_variable('weight_stddev', shape=(num_inputs, num_outputs),
                                            initializer=kernel_stddev_initializer)
            bias_stddev = tf.get_variable('bias_stddev', shape=(num_outputs,),
                                          initializer=kernel_stddev_initializer)
            bias_noise = _f(tf.random_normal((num_outputs,)))
            weight_noise = _factorized_noise(num_inputs, num_outputs)

        bias = bias_mean + bias_stddev * bias_noise
        weight = weight_mean + weight_stddev * weight_noise
        return activation(tf.matmul(input_layer, weight) + bias)

    @staticmethod
    @reg_to_tf(layers.NoisyNetDense)
    def to_tf(base: layers.NoisyNetDense):
        return NoisyNetDense(units=base.units)
