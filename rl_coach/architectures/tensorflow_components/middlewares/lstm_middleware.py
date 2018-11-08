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


import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import batchnorm_activation_dropout, Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_LSTM_Embedding
from rl_coach.utils import force_list


class LSTMMiddleware(Middleware):
    def __init__(self, activation_function=tf.nn.relu, number_of_lstm_cells: int=256,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_lstm_embedder", dense_layer=Dense, is_training=False):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout_rate=dropout_rate, scheme=scheme, name=name, dense_layer=dense_layer,
                         is_training=is_training)
        self.return_type = Middleware_LSTM_Embedding
        self.number_of_lstm_cells = number_of_lstm_cells
        self.layers = []

    def _build_module(self):
        """
        self.state_in: tuple of placeholders containing the initial state
        self.state_out: tuple of output state

        todo: it appears that the shape of the output is batch, feature
        the code here seems to be slicing off the first element in the batch
        which would definitely be wrong. need to double check the shape
        """

        self.layers.append(self.input)

        # optionally insert some layers before the LSTM
        for idx, layer_params in enumerate(self.layers_params):
            self.layers.extend(force_list(
                layer_params(self.layers[-1], name='fc{}'.format(idx),
                             is_training=self.is_training)
            ))

        # add the LSTM layer
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.number_of_lstm_cells, state_is_tuple=True)
        self.c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        self.h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [self.c_init, self.h_init]
        self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (self.c_in, self.h_in)
        rnn_in = tf.expand_dims(self.layers[-1], [0])
        step_size = tf.shape(self.layers[-1])[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_in, self.h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        self.output = tf.reshape(lstm_outputs, [-1, self.number_of_lstm_cells])

    @property
    def schemes(self):
        return {
            MiddlewareScheme.Empty:
                [],

            # ppo
            MiddlewareScheme.Shallow:
                [
                    self.dense_layer(64)
                ],

            # dqn
            MiddlewareScheme.Medium:
                [
                    self.dense_layer(512)
                ],

            MiddlewareScheme.Deep: \
                [
                    self.dense_layer(128),
                    self.dense_layer(128),
                    self.dense_layer(128)
                ]
        }

