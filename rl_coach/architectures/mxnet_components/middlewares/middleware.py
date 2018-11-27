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

from typing import Union
from types import ModuleType

import mxnet as mx
from mxnet.gluon import nn
from rl_coach.architectures.middleware_parameters import MiddlewareParameters
from rl_coach.architectures.mxnet_components.layers import convert_layer
from rl_coach.base_parameters import MiddlewareScheme

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class Middleware(nn.HybridBlock):
    def __init__(self, params: MiddlewareParameters):
        """
        Middleware is the middle part of the network. It takes the embeddings from the input embedders,
        after they were aggregated in some method (for example, concatenation) and passes it through a neural network
        which can be customizable but shared between the heads of the network.

        :param params: parameters object containing batchnorm, activation_function and dropout properties.
        """
        super(Middleware, self).__init__()
        self.scheme = params.scheme

        with self.name_scope():
            self.net = nn.HybridSequential()
            if isinstance(self.scheme, MiddlewareScheme):
                blocks = self.schemes[self.scheme]
            else:
                # if scheme is specified directly, convert to MX layer if it's not a callable object
                # NOTE: if layer object is callable, it must return a gluon block when invoked
                blocks = [convert_layer(l) for l in self.scheme]
            for block in blocks:
                self.net.add(block())
                if params.batchnorm:
                    self.net.add(nn.BatchNorm())
                if params.activation_function:
                    self.net.add(nn.Activation(params.activation_function))
                if params.dropout_rate:
                    self.net.add(nn.Dropout(rate=params.dropout_rate))

    @property
    def schemes(self) -> dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used for the
        Middleware. Should be implemented in child classes, and are used to create Block when Middleware is initialised.

        :return: dictionary of schemes, with key of type MiddlewareScheme enum and value being list of mxnet.gluon.Block.
        """
        raise NotImplementedError("Inheriting embedder must define schemes matching its allowed default "
                                  "configurations.")

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through middleware network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: state embedding, of shape (batch_size, in_channels).
        :return: state middleware embedding, where shape is (batch_size, channels).
        """
        return self.net(x)
