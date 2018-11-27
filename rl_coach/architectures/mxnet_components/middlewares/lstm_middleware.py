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
Module that defines the LSTM middleware class
"""

from typing import Union
from types import ModuleType

import mxnet as mx
from mxnet.gluon import rnn
from rl_coach.architectures.mxnet_components.layers import Dense
from rl_coach.architectures.mxnet_components.middlewares.middleware import Middleware
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import MiddlewareScheme

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class LSTMMiddleware(Middleware):
    def __init__(self, params: LSTMMiddlewareParameters):
        """
        LSTMMiddleware or Long Short Term Memory Middleware can be used in the middle part of the network. It takes the
        embeddings from the input embedders, after they were aggregated in some method (for example, concatenation)
        and passes it through a neural network  which can be customizable but shared between the heads of the network.

        :param params: parameters object containing batchnorm, activation_function, dropout and
            number_of_lstm_cells properties.
        """
        super(LSTMMiddleware, self).__init__(params)
        self.number_of_lstm_cells = params.number_of_lstm_cells
        with self.name_scope():
            self.lstm = rnn.LSTM(hidden_size=self.number_of_lstm_cells)

    @property
    def schemes(self) -> dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used for the
        Middleware. Are used to create Block when LSTMMiddleware is initialised, and are applied before the LSTM.

        :return: dictionary of schemes, with key of type MiddlewareScheme enum and value being list of mxnet.gluon.Block.
        """
        return {
            MiddlewareScheme.Empty:
                [],

            # Use for PPO
            MiddlewareScheme.Shallow:
                [
                    Dense(units=64)
                ],

            # Use for DQN
            MiddlewareScheme.Medium:
                [
                    Dense(units=512)
                ],

            MiddlewareScheme.Deep:
                [
                    Dense(units=128),
                    Dense(units=128),
                    Dense(units=128)
                ]
        }

    def hybrid_forward(self,
                       F: ModuleType,
                       x: nd_sym_type,
                       *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through LSTM middleware network.
        Applies dense layers from selected scheme before passing result to LSTM layer.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: state embedding, of shape (batch_size, in_channels).
        :return: state middleware embedding, where shape is (batch_size, channels).
        """
        x_ntc = x.reshape(shape=(0, 0, -1))
        emb_ntc = super(LSTMMiddleware, self).hybrid_forward(F, x_ntc, *args, **kwargs)
        emb_tnc = emb_ntc.transpose(axes=(1, 0, 2))
        return self.lstm(emb_tnc)
