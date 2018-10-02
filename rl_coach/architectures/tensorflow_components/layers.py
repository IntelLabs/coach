import math
from typing import List, Union

import tensorflow as tf

from rl_coach.utils import force_list


def batchnorm_activation_dropout(input_layer, batchnorm, activation_function, dropout, dropout_rate, is_training, name):
    layers = [input_layer]

    # batchnorm
    if batchnorm:
        layers.append(
            tf.layers.batch_normalization(layers[-1], name="{}_batchnorm".format(name), training=is_training)
        )

    # activation
    if activation_function:
        layers.append(
            activation_function(layers[-1], name="{}_activation".format(name))
        )

    # dropout
    if dropout:
        layers.append(
            tf.layers.dropout(layers[-1], dropout_rate, name="{}_dropout".format(name), training=is_training)
        )

    # remove the input layer from the layers list
    del layers[0]

    return layers


class Conv2d(object):
    def __init__(self, num_filters: int, kernel_size: int, strides: int):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides

    def __call__(self, input_layer, name: str=None, is_training=None):
        """
        returns a tensorflow conv2d layer
        :param input_layer: previous layer
        :param name: layer name
        :return: conv2d layer
        """
        return tf.layers.conv2d(input_layer, filters=self.num_filters, kernel_size=self.kernel_size,
                                strides=self.strides, data_format='channels_last', name=name)

    def __str__(self):
        return "Convolution (num filters = {}, kernel size = {}, stride = {})"\
            .format(self.num_filters, self.kernel_size, self.strides)


class BatchnormActivationDropout(object):
    def __init__(self, batchnorm: bool=False, activation_function=None, dropout_rate: float=0):
        self.batchnorm = batchnorm
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

    def __call__(self, input_layer, name: str=None, is_training=None):
        """
        returns a list of tensorflow batchnorm, activation and dropout layers
        :param input_layer: previous layer
        :param name: layer name
        :return: batchnorm, activation and dropout layers
        """
        return batchnorm_activation_dropout(input_layer, batchnorm=self.batchnorm,
                                            activation_function=self.activation_function,
                                            dropout=self.dropout_rate > 0, dropout_rate=self.dropout_rate,
                                            is_training=is_training, name=name)

    def __str__(self):
        result = []
        if self.batchnorm:
            result += ["Batch Normalization"]
        if self.activation_function:
            result += ["Activation (type = {})".format(self.activation_function.__name__)]
        if self.dropout_rate > 0:
            result += ["Dropout (rate = {})".format(self.dropout_rate)]
        return "\n".join(result)


class Dense(object):
    def __init__(self, units: int):
        self.units = units

    def __call__(self, input_layer, name: str=None, kernel_initializer=None, activation=None, is_training=None):
        """
        returns a tensorflow dense layer
        :param input_layer: previous layer
        :param name: layer name
        :return: dense layer
        """
        return tf.layers.dense(input_layer, self.units, name=name, kernel_initializer=kernel_initializer,
                               activation=activation)

    def __str__(self):
        return "Dense (num outputs = {})".format(self.units)


class NoisyNetDense(object):
    """
    A factorized Noisy Net layer

    https://arxiv.org/abs/1706.10295.
    """

    def __init__(self, units: int):
        self.units = units
        self.sigma0 = 0.5

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
            bias_noise = self.f(tf.random_normal((num_outputs,)))
            weight_noise = self.factorized_noise(num_inputs, num_outputs)

        bias = bias_mean + bias_stddev * bias_noise
        weight = weight_mean + weight_stddev * weight_noise
        return activation(tf.matmul(input_layer, weight) + bias)

    def factorized_noise(self, inputs, outputs):
        # TODO: use factorized noise only for compute intensive algos (e.g. DQN).
        #      lighter algos (e.g. DQN) should not use it
        noise1 = self.f(tf.random_normal((inputs, 1)))
        noise2 = self.f(tf.random_normal((1, outputs)))
        return tf.matmul(noise1, noise2)

    @staticmethod
    def f(values):
        return tf.sqrt(tf.abs(values)) * tf.sign(values)

    def __str__(self):
        return "Noisy Dense (num outputs = {})".format(self.units)
