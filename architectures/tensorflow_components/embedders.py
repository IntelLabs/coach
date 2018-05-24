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
from configurations import EmbedderDepth, EmbedderWidth


class InputEmbedder(object):
    def __init__(self, input_size, activation_function=tf.nn.relu,
                 embedder_depth=EmbedderDepth.Shallow, embedder_width=EmbedderWidth.Wide,
                 name="embedder"):
        self.name = name
        self.input_size = input_size
        self.activation_function = activation_function
        self.input = None
        self.output = None
        self.embedder_depth = embedder_depth
        self.embedder_width = embedder_width

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
    def __init__(self, input_size, input_rescaler=255.0, activation_function=tf.nn.relu,
                 embedder_depth=EmbedderDepth.Shallow, embedder_width=EmbedderWidth.Wide,
                 name="embedder"):
        InputEmbedder.__init__(self, input_size, activation_function, embedder_depth, embedder_width, name)
        self.input_rescaler = input_rescaler

    def _build_module(self):
        # image observation
        rescaled_observation_stack = self.input / self.input_rescaler

        if self.embedder_depth == EmbedderDepth.Shallow:
            # same embedder as used in the original DQN paper
            self.observation_conv1 = tf.layers.conv2d(rescaled_observation_stack,
                                                      filters=32, kernel_size=(8, 8), strides=(4, 4),
                                                      activation=self.activation_function, data_format='channels_last',
                                                      name='conv1')
            self.observation_conv2 = tf.layers.conv2d(self.observation_conv1,
                                                      filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                      activation=self.activation_function, data_format='channels_last',
                                                      name='conv2')
            self.observation_conv3 = tf.layers.conv2d(self.observation_conv2,
                                                      filters=64, kernel_size=(3, 3), strides=(1, 1),
                                                      activation=self.activation_function, data_format='channels_last',
                                                      name='conv3'
                                                      )

            self.output = tf.contrib.layers.flatten(self.observation_conv3)

        elif self.embedder_depth == EmbedderDepth.Deep:
            # the embedder used in the CARLA papers
            self.observation_conv1 = tf.layers.conv2d(rescaled_observation_stack,
                                                 filters=32, kernel_size=(5, 5), strides=(2, 2),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv1')
            self.observation_conv2 = tf.layers.conv2d(self.observation_conv1,
                                                 filters=32, kernel_size=(3, 3), strides=(1, 1),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv2')
            self.observation_conv3 = tf.layers.conv2d(self.observation_conv2,
                                                 filters=64, kernel_size=(3, 3), strides=(2, 2),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv3')
            self.observation_conv4 = tf.layers.conv2d(self.observation_conv3,
                                                 filters=64, kernel_size=(3, 3), strides=(1, 1),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv4')
            self.observation_conv5 = tf.layers.conv2d(self.observation_conv4,
                                                 filters=128, kernel_size=(3, 3), strides=(2, 2),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv5')
            self.observation_conv6 = tf.layers.conv2d(self.observation_conv5,
                                                 filters=128, kernel_size=(3, 3), strides=(1, 1),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv6')
            self.observation_conv7 = tf.layers.conv2d(self.observation_conv6,
                                                 filters=256, kernel_size=(3, 3), strides=(2, 2),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv7')
            self.observation_conv8 = tf.layers.conv2d(self.observation_conv7,
                                                 filters=256, kernel_size=(3, 3), strides=(1, 1),
                                                 activation=self.activation_function, data_format='channels_last',
                                                 name='conv8')

            self.output = tf.contrib.layers.flatten(self.observation_conv8)
        else:
            raise ValueError("The defined embedder complexity value is invalid")


class VectorEmbedder(InputEmbedder):
    def __init__(self, input_size, activation_function=tf.nn.relu,
                 embedder_depth=EmbedderDepth.Shallow, embedder_width=EmbedderWidth.Wide,
                 name="embedder"):
        InputEmbedder.__init__(self, input_size, activation_function, embedder_depth, embedder_width, name)

    def _build_module(self):
        # vector observation
        input_layer = tf.contrib.layers.flatten(self.input)

        width = 128 if self.embedder_width == EmbedderWidth.Wide else 32

        if self.embedder_depth == EmbedderDepth.Shallow:
            self.output = tf.layers.dense(input_layer, 2*width, activation=self.activation_function,
                                                 name='fc1')

        elif self.embedder_depth == EmbedderDepth.Deep:
            # the embedder used in the CARLA papers
            self.observation_fc1 = tf.layers.dense(input_layer, width, activation=self.activation_function,
                                                 name='fc1')
            self.observation_fc2 = tf.layers.dense(self.observation_fc1, width, activation=self.activation_function,
                                                 name='fc2')
            self.output = tf.layers.dense(self.observation_fc2, width, activation=self.activation_function,
                                                 name='fc3')
        else:
            raise ValueError("The defined embedder complexity value is invalid")
