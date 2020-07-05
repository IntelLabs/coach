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

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_LSTM_Embedding


class LSTMMiddleware(Middleware):
    def __init__(self, activation_function=tf.nn.relu, number_of_lstm_cells: int=256,
                 sequence_length: int=-1, stride: int=-1,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_lstm_embedder", dense_layer=Dense, is_training=False):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout_rate=dropout_rate, scheme=scheme, name=name, dense_layer=dense_layer,
                         is_training=is_training)
        self.return_type = Middleware_LSTM_Embedding
        self.number_of_lstm_cells = number_of_lstm_cells
        self.sequence_length = sequence_length
        self.stride = stride
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
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.number_of_lstm_cells, state_is_tuple=True,
                                           activation=self.activation_function)
        self.c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        self.h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        if self.sequence_length == -1:
            rnn_in = tf.expand_dims(self.layers[-1], [0])
        else:
            seq_len = tf.cond(tf.equal(tf.shape(self.layers[-1])[0], tf.constant(1)),
                              lambda: 1, lambda: self.sequence_length)
            rnn_in = tf.reshape(self.layers[-1], [-1, seq_len] + list(self.layers[-1].shape[1:]))
        batch_size = tf.shape(rnn_in)[0]
        self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(self.c_in, (batch_size, 1)),
                                                 tf.tile(self.h_in, (batch_size, 1)))
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, time_major=False)

        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c, lstm_h)
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

