from typing import Union
from types import ModuleType

import mxnet as mx
from mxnet.gluon import nn
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.mxnet_components.layers import convert_layer
from rl_coach.base_parameters import EmbedderScheme

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class InputEmbedder(nn.HybridBlock):
    def __init__(self, params: InputEmbedderParameters):
        """
        An input embedder is the first part of the network, which takes the input from the state and produces a vector
        embedding by passing it through a neural network. The embedder will mostly be input type dependent, and there
        can be multiple embedders in a single network.

        :param params: parameters object containing input_clipping, input_rescaling, batchnorm, activation_function
            and dropout properties.
        """
        super(InputEmbedder, self).__init__()
        self.embedder_name = params.name
        self.input_clipping = params.input_clipping
        self.scheme = params.scheme

        with self.name_scope():
            self.net = nn.HybridSequential()
            if isinstance(self.scheme, EmbedderScheme):
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
        InputEmbedder. Should be implemented in child classes, and are used to create Block when InputEmbedder is
        initialised.

        :return: dictionary of schemes, with key of type EmbedderScheme enum and value being list of mxnet.gluon.Block.
        """
        raise NotImplementedError("Inheriting embedder must define schemes matching its allowed default "
                                  "configurations.")

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through embedder network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: environment state, where first dimension is batch_size, then dimensions are data type dependent.
        :return: embedding of environment state, where shape is (batch_size, channels).
        """
        # `input_rescaling` and `input_offset` set on inheriting embedder
        x = x / self.input_rescaling
        x = x - self.input_offset
        if self.input_clipping is not None:
            x.clip(a_min=self.input_clipping[0], a_max=self.input_clipping[1])
        x = self.net(x)
        return x.flatten()
