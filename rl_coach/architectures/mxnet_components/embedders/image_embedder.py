from typing import Union
from types import ModuleType

import mxnet as mx
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.mxnet_components.embedders.embedder import InputEmbedder
from rl_coach.architectures.mxnet_components.layers import Conv2d
from rl_coach.base_parameters import EmbedderScheme

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class ImageEmbedder(InputEmbedder):
    def __init__(self, params: InputEmbedderParameters):
        """
        An image embedder is an input embedder that takes an image input from the state and produces a vector
        embedding by passing it through a neural network.

        :param params: parameters object containing input_clipping, input_rescaling, batchnorm, activation_function
            and dropout properties.
        """
        super(ImageEmbedder, self).__init__(params)
        self.input_rescaling = params.input_rescaling['image']
        self.input_offset = params.input_offset['image']

    @property
    def schemes(self) -> dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used. Are used
        to create Block when ImageEmbedder is initialised.

        :return: dictionary of schemes, with key of type EmbedderScheme enum and value being list of mxnet.gluon.Block.
        """
        return {
            EmbedderScheme.Empty:
                [],

            EmbedderScheme.Shallow:
                [
                    Conv2d(num_filters=32, kernel_size=8, strides=4)
                ],

            # Use for Atari DQN
            EmbedderScheme.Medium:
                [
                    Conv2d(num_filters=32, kernel_size=8, strides=4),
                    Conv2d(num_filters=64, kernel_size=4, strides=2),
                    Conv2d(num_filters=64, kernel_size=3, strides=1)
                ],

            # Use for Carla
            EmbedderScheme.Deep:
                [
                    Conv2d(num_filters=32, kernel_size=5, strides=2),
                    Conv2d(num_filters=32, kernel_size=3, strides=1),
                    Conv2d(num_filters=64, kernel_size=3, strides=2),
                    Conv2d(num_filters=64, kernel_size=3, strides=1),
                    Conv2d(num_filters=128, kernel_size=3, strides=2),
                    Conv2d(num_filters=128, kernel_size=3, strides=1),
                    Conv2d(num_filters=256, kernel_size=3, strides=2),
                    Conv2d(num_filters=256, kernel_size=3, strides=1)
                ]
        }

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, *args, **kwargs) -> nd_sym_type:
        """
        Used for forward pass through embedder network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: image representing environment state, of shape (batch_size, in_channels, height, width).
        :return: embedding of environment state, of shape (batch_size, channels).
        """
        # convert from NHWC to NCHW (default for MXNet Convolutions)
        x = x.transpose((0,3,1,2))
        return super(ImageEmbedder, self).hybrid_forward(F, x, *args, **kwargs)
