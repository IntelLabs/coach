from typing import Union
from types import ModuleType

import mxnet as mx
from mxnet import nd, sym
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.mxnet_components.embedders.embedder import InputEmbedder
from rl_coach.architectures.mxnet_components.layers import Dense
from rl_coach.base_parameters import EmbedderScheme

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class VectorEmbedder(InputEmbedder):
    def __init__(self, params: InputEmbedderParameters):
        """
        An vector embedder is an input embedder that takes an vector input from the state and produces a vector
        embedding by passing it through a neural network.

        :param params: parameters object containing input_clipping, input_rescaling, batchnorm, activation_function
            and dropout properties.
        """
        super(VectorEmbedder, self).__init__(params)
        self.input_rescaling = params.input_rescaling['vector']
        self.input_offset = params.input_offset['vector']

    @property
    def schemes(self):
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used. Are used
        to create Block when VectorEmbedder is initialised.

        :return: dictionary of schemes, with key of type EmbedderScheme enum and value being list of mxnet.gluon.Block.
        """
        return {
            EmbedderScheme.Empty:
                [],

            EmbedderScheme.Shallow:
                [
                    Dense(units=128)
                ],

            # Use for DQN
            EmbedderScheme.Medium:
                [
                    Dense(units=256)
                ],

            # Use for Carla
            EmbedderScheme.Deep:
                [
                    Dense(units=128),
                    Dense(units=128),
                    Dense(units=128)
                ]
        }

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through embedder network.

        :param F: backend api, either `nd` or `sym` (if block has been hybridized).
        :type F: nd or sym
        :param x: vector representing environment state, of shape (batch_size, in_channels).
        :return: embedding of environment state, of shape (batch_size, channels).
        """
        if isinstance(x, nd.NDArray) and len(x.shape) != 2 and self.scheme != EmbedderScheme.Empty:
            raise ValueError("Vector embedders expect the input size to have 2 dimensions. The given size is: {}"
                             .format(x.shape))
        return super(VectorEmbedder, self).hybrid_forward(F, x, *args, **kwargs)
