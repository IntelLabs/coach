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
from tensorflow import keras


from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class Head(keras.layers.Layer):
    """
    A head is the final part of the network. It takes the embedding from the middleware embedder and passes it through
    a neural network to produce the output of the network. There can be multiple heads in a network, and each one has
    an assigned loss function. The heads are algorithm dependent.

    :param agent_parameters: containing algorithm parameters such as clip_likelihood_ratio_using_epsilon
        and beta_entropy.
    :param spaces: con e: name of head network. currently unused.
    :param head_type_idx: index of head network. currently unused.
    :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
    :param is_local: flag to denote if network is local. currently unused.
    :param activation_function: activation function to use between layers. currently unused.
    """

    def __init__(self,
                 # agent_parameters: AgentParameters,
                 # spaces: SpacesDefinition,
                 # network_name: str,
                 # head_idx: int = 0,
                 # loss_weight: float=1.,
                 # is_local: bool = True,
                 # activation_function=tf.nn.relu,
                 **kwargs):
        super().__init__(**kwargs)
        # self.head_idx = head_idx
        # self.network_name = network_name
        # self.loss = []
        # self.loss_type = []
        # self.regularizations = []
        # self.target = []
        # self.importance_weight = []
        # self.is_local = is_local
        # self.ap = agent_parameters
        # self.spaces = spaces
        # self.return_type = None


    def loss(self):
        """
        The loss had moved to a separate class.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return '{}_{}'.format(self.name, self.head_idx)

    @classmethod
    def path(cls):
        return cls.__class__.__name__
