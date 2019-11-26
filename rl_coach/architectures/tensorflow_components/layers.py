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
import tensorflow as tf
from tensorflow import keras

from rl_coach.architectures import layers
from rl_coach.architectures.tensorflow_components import utils



def batchnorm_activation_dropout(input_layer, batchnorm, activation_function, dropout_rate, is_training, name):
    layers = [input_layer]

    # Rationale: passing a bool here will mean that batchnorm and or activation will never activate
    assert not isinstance(is_training, bool)

    # batchnorm
    if batchnorm:

        layers.append(
            tf.compat.v1.layers.batch_normalization(layers[-1], name="{}_batchnorm".format(name), training=is_training)
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
            tf.compat.v1.layers.dropout(layers[-1], dropout_rate, name="{}_dropout".format(name), training=is_training)
        )

    # remove the input layer from the layers list
    del layers[0]

    return layers


# define global dictionary for storing layer type to layer implementation mapping
tf_layer_dict = dict()


def reg_to_tf_instance(layer_type) -> FunctionType:
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
    If layer instance is callable (meaning this is already a concrete TF class), return layer, otherwise convert to TF type
    :param layer: layer to be converted
    :return: converted layer if not callable, otherwise layer itself
    """
    if callable(layer):
        return layer
    return tf_layer_dict[type(layer)](layer)


# def convert_layer_class(layer_class):
#     """
#     If layer instance is callable, return layer, otherwise convert to TF type
#     :param layer: layer to be converted
#     :return: converted layer if not callable, otherwise layer itself
#     """
#     if hasattr(layer_class, 'to_tf_instance'):
#         return layer_class
#     else:
#         return tf_layer_class_dict[layer_class]()


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
        # return tf.compat.v1.layers.conv2d(input_layer, filters=self.num_filters, kernel_size=self.kernel_size,
        #                         strides=self.strides, data_format='channels_last', name=name)

        return tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=self.kernel_size,
                                          strides=self.strides, data_format='channels_last', name=name)


    @staticmethod
    @reg_to_tf_instance(layers.Conv2d)
    def to_tf_instance(base: layers.Conv2d):
            return Conv2d(
                num_filters=base.num_filters,
                kernel_size=base.kernel_size,
                strides=base.strides)

    # @staticmethod
    # @reg_to_tf_class(layers.Conv2d)
    # def to_tf_class():
    #     return Conv2d


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
    @reg_to_tf_instance(layers.BatchnormActivationDropout)
    def to_tf_instance(base: layers.BatchnormActivationDropout):
        return BatchnormActivationDropout, BatchnormActivationDropout(
                batchnorm=base.batchnorm,
                activation_function=base.activation_function,
                dropout_rate=base.dropout_rate)

    # @staticmethod
    # @reg_to_tf_class(layers.BatchnormActivationDropout)
    # def to_tf_class():
    #     return BatchnormActivationDropout


# class Dense(keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         super().__init__(**kwargs)
#         self.dense = tf.keras.layers.Dense(units)
#
#     def call(self, inputs):
#         return self.dense(inputs)

class Dense(layers.Dense):
    def __init__(self, units: int):
        super(Dense, self).__init__(units=units)

    def __call__(self):
        """
        returns a tensorflow dense layer
        :return: dense layer
        """
        return tf.keras.layers.Dense(self.units)

    @staticmethod
    @reg_to_tf_instance(layers.Dense)
    def to_tf_instance(base: layers.Dense):
        return Dense(units=base.units)()





#
# class Dense(layers.Dense):
#     def __init__(self, units: int):
#         super(Dense, self).__init__(units=units)
#
#     def __call__(self, input_layer, name: str=None, kernel_initializer=None, activation=None, is_training=None):
#         """
#         returns a tensorflow dense layer
#         :param input_layer: previous layer
#         :param name: layer name
#         :return: dense layer
#         """
#         return tf.compat.v1.layers.dense(input_layer, self.units, name=name, kernel_initializer=kernel_initializer,
#                                activation=activation)
#
#     @staticmethod
#     @reg_to_tf_instance(layers.Dense)
#     def to_tf_instance(base: layers.Dense):
#         return Dense(units=base.units)
#
#     @staticmethod
#     @reg_to_tf_class(layers.Dense)
#     def to_tf_class():
#         return Dense
#

