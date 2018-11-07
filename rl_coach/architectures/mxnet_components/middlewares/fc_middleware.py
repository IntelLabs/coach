"""
Module that defines the fully-connected middleware class
"""

from rl_coach.architectures.mxnet_components.layers import Dense
from rl_coach.architectures.mxnet_components.middlewares.middleware import Middleware
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import MiddlewareScheme


class FCMiddleware(Middleware):
    def __init__(self, params: FCMiddlewareParameters):
        """
        FCMiddleware or Fully-Connected Middleware can be used in the middle part of the network. It takes the
        embeddings from the input embedders, after they were aggregated in some method (for example, concatenation)
        and passes it through a neural network  which can be customizable but shared between the heads of the network.

        :param params: parameters object containing batchnorm, activation_function and dropout properties.
        """
        super(FCMiddleware, self).__init__(params)

    @property
    def schemes(self) -> dict:
        """
        Schemes are the pre-defined network architectures of various depths and complexities that can be used for the
        Middleware. Are used to create Block when FCMiddleware is initialised.

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
