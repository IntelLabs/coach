from typing import Union
from types import ModuleType

import mxnet as mx
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.mxnet_components.embedders.embedder import InputEmbedder

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class TensorEmbedder(InputEmbedder):
    def __init__(self, params: InputEmbedderParameters):
        """
        A tensor embedder is an input embedder that takes a tensor with arbitrary dimension and produces a vector
        embedding by passing it through a neural network. An example is video data or 3D image data (i.e. 4D tensors)
    or other type of data that is more than 1 dimension (i.e. not vector) but is not an image.

        NOTE: There are no pre-defined schemes for tensor embedder. User must define a custom scheme by passing
        a callable object as InputEmbedderParameters.scheme when defining the respective preset. This callable
        object must return a Gluon HybridBlock. The hybrid_forward() of this block must accept a single input,
        normalized observation, and return an embedding vector for each sample in the batch.
        Keep in mind that the scheme is a list of blocks, which are stacked by optional batchnorm,
        activation, and dropout in between as specified in InputEmbedderParameters.

        :param params: parameters object containing input_clipping, input_rescaling, batchnorm, activation_function
            and dropout properties.
        """
        super(TensorEmbedder, self).__init__(params)
        self.input_rescaling = params.input_rescaling['tensor']
        self.input_offset = params.input_offset['tensor']

    @property
    def schemes(self) -> dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used. Are used
        to create Block when InputEmbedder is initialised.

        Note: Tensor embedder doesn't define any pre-defined scheme. User must provide custom scheme in preset.

        :return: dictionary of schemes, with key of type EmbedderScheme enum and value being list of mxnet.gluon.Block.
            For tensor embedder, this is an empty dictionary.
        """
        return {}

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through embedder network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: image representing environment state, of shape (batch_size, in_channels, height, width).
        :return: embedding of environment state, of shape (batch_size, channels).
        """
        return super(TensorEmbedder, self).hybrid_forward(F, x, *args, **kwargs)
