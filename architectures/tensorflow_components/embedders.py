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


class InputEmbedder(object):
    def __init__(self, input_size, activation_function=tf.nn.relu, name="embedder"):
        self.name = name
        self.input_size = input_size
        self.activation_function = activation_function
        self.input = None
        self.output = None

    def __call__(self, prev_input_placeholder=None):
        with tf.variable_scope(self.get_name()):
            if prev_input_placeholder is None:
                self.input = tf.placeholder("float", shape=(None,) + self.input_size, name=self.get_name())
            else:
                self.input = prev_input_placeholder
            self._build_module()

        return self.input, self.output

    def _build_module(self):
        pass

    def get_name(self):
        return self.name


class ImageEmbedder(InputEmbedder):
    def __init__(self, input_size, input_rescaler=255.0, activation_function=tf.nn.relu, name="embedder"):
        InputEmbedder.__init__(self, input_size, activation_function, name)
        self.input_rescaler = input_rescaler

    def _build_module(self):
        # image observation
        rescaled_observation_stack = self.input / self.input_rescaler
        self.observation_conv1 = tf.layers.conv2d(rescaled_observation_stack,
                                             filters=32, kernel_size=(8, 8), strides=(4, 4),
                                             activation=self.activation_function, data_format='channels_last')
        self.observation_conv2 = tf.layers.conv2d(self.observation_conv1,
                                             filters=64, kernel_size=(4, 4), strides=(2, 2),
                                             activation=self.activation_function, data_format='channels_last')
        self.observation_conv3 = tf.layers.conv2d(self.observation_conv2,
                                             filters=64, kernel_size=(3, 3), strides=(1, 1),
                                             activation=self.activation_function, data_format='channels_last')

        self.output = tf.contrib.layers.flatten(self.observation_conv3)


class VectorEmbedder(InputEmbedder):
    def __init__(self, input_size, activation_function=tf.nn.relu, name="embedder"):
        InputEmbedder.__init__(self, input_size, activation_function, name)

    def _build_module(self):
        # vector observation
        input_layer = tf.contrib.layers.flatten(self.input)
        self.output = tf.layers.dense(input_layer, 256, activation=self.activation_function)
