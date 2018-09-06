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
from typing import Type

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction
from rl_coach.architectures.tensorflow_components.architecture import Dense
from rl_coach.base_parameters import AgentParameters, Parameters, NetworkComponentParameters
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class HeadParameters(NetworkComponentParameters):
    def __init__(self, parameterized_class: Type['Head'], activation_function: str = 'relu', name: str= 'head',
                 dense_layer=Dense):
        super().__init__(dense_layer=dense_layer)
        self.activation_function = activation_function
        self.name = name
        self.parameterized_class_name = parameterized_class.__name__


class Head(object):
    """
    A head is the final part of the network. It takes the embedding from the middleware embedder and passes it through
    a neural network to produce the output of the network. There can be multiple heads in a network, and each one has
    an assigned loss function. The heads are algorithm dependent.
    """
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int=0, loss_weight: float=1., is_local: bool=True, activation_function: str='relu',
                 dense_layer=Dense):
        self.head_idx = head_idx
        self.network_name = network_name
        self.network_parameters = agent_parameters.network_wrappers[self.network_name]
        self.name = "head"
        self.output = []
        self.loss = []
        self.loss_type = []
        self.regularizations = []
        self.loss_weight = tf.Variable([float(w) for w in force_list(loss_weight)],
                                       trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.loss_weight_placeholder = tf.placeholder("float")
        self.set_loss_weight = tf.assign(self.loss_weight, self.loss_weight_placeholder)
        self.target = []
        self.importance_weight = []
        self.input = []
        self.is_local = is_local
        self.ap = agent_parameters
        self.spaces = spaces
        self.return_type = None
        self.activation_function = activation_function
        self.dense_layer = dense_layer

    def __call__(self, input_layer):
        """
        Wrapper for building the module graph including scoping and loss creation
        :param input_layer: the input to the graph
        :return: the output of the last layer and the target placeholder
        """
        with tf.variable_scope(self.get_name(), initializer=tf.contrib.layers.xavier_initializer()):
            self._build_module(input_layer)

            self.output = force_list(self.output)
            self.target = force_list(self.target)
            self.input = force_list(self.input)
            self.loss_type = force_list(self.loss_type)
            self.loss = force_list(self.loss)
            self.regularizations = force_list(self.regularizations)
            if self.is_local:
                self.set_loss()
            self._post_build()

        if self.is_local:
            return self.output, self.target, self.input, self.importance_weight
        else:
            return self.output, self.input

    def _build_module(self, input_layer):
        """
        Builds the graph of the module
        This method is called early on from __call__. It is expected to store the graph
        in self.output.
        :param input_layer: the input to the graph
        :return: None
        """
        pass

    def _post_build(self):
        """
        Optional function that allows adding any extra definitions after the head has been fully defined
        For example, this allows doing additional calculations that are based on the loss
        :return: None
        """
        pass

    def get_name(self):
        """
        Get a formatted name for the module
        :return: the formatted name
        """
        return '{}_{}'.format(self.name, self.head_idx)

    def set_loss(self):
        """
        Creates a target placeholder and loss function for each loss_type and regularization
        :param loss_type: a tensorflow loss function
        :param scope: the name scope to include the tensors in
        :return: None
        """

        # there are heads that define the loss internally, but we need to create additional placeholders for them
        for idx in range(len(self.loss)):
            importance_weight = tf.placeholder('float',
                                               [None] + [1] * (len(self.target[idx].shape) - 1),
                                               '{}_importance_weight'.format(self.get_name()))
            self.importance_weight.append(importance_weight)

        # add losses and target placeholder
        for idx in range(len(self.loss_type)):
            # create target placeholder
            target = tf.placeholder('float', self.output[idx].shape, '{}_target'.format(self.get_name()))
            self.target.append(target)

            # create importance sampling weights placeholder
            num_target_dims = len(self.target[idx].shape)
            importance_weight = tf.placeholder('float', [None] + [1] * (num_target_dims - 1),
                                               '{}_importance_weight'.format(self.get_name()))
            self.importance_weight.append(importance_weight)

            # compute the weighted loss. importance_weight weights over the samples in the batch, while self.loss_weight
            # weights the specific loss of this head against other losses in this head or in other heads
            loss_weight = self.loss_weight[idx]*importance_weight
            loss = self.loss_type[idx](self.target[-1], self.output[idx],
                                       scope=self.get_name(), reduction=Reduction.NONE, loss_collection=None)

            # the loss is first summed over each sample in the batch and then the mean over the batch is taken
            loss = tf.reduce_mean(loss_weight*tf.reduce_sum(loss, axis=list(range(1, num_target_dims))))

            # we add the loss to the losses collection and later we will extract it in general_network
            tf.losses.add_loss(loss)
            self.loss.append(loss)

        # add regularizations
        for regularization in self.regularizations:
            self.loss.append(regularization)

    @classmethod
    def path(cls):
        return cls.__class__.__name__
