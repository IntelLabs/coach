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

from typing import List, Tuple, Union


LOSS_OUT_TYPE_KL = 'kl_divergence'
LOSS_OUT_TYPE_ENTROPY = 'entropy'
LOSS_OUT_TYPE_LIKELIHOOD_RATIO = 'likelihood_ratio'
LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO = 'clipped_likelihood_ratio'

from tensorflow import keras
from tensorflow import Tensor

import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head#, normalized_columns_initializer
from rl_coach.base_parameters import AgentParameters, DistributedTaskParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace
from rl_coach.spaces import SpacesDefinition

from rl_coach.utils import eps

class ContinuousPPOHead(keras.layers.Layer):
    def __init__(self, num_actions: int) -> None:
        """
        Head block for Continuous Proximal Policy Optimization, to calculate probabilities for each action given
        middleware representation of the environment state.

        :param num_actions: number of actions in action space.
        """

        super(ContinuousPPOHead, self).__init__()
        # all samples (across batch, and time step) share the same covariance, which is learnt,
        # but since we assume the action probability variables are independent,
        # only the diagonal entries of the covariance matrix are specified.
        self.policy_means_layer = tf.keras.layers.Dense(units=num_actions)
        # self.policy_log_std_layer = tf.keras.layers.Dense(units=num_actions,
        #                                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=None),
        #                                                   bias_initializer='zeros')

        self.policy_log_std_layer = tf.Variable(tf.zeros((1, num_actions)), dtype='float32', name="policy_log_std")
        #self.action_proba = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[..., 0], scale_diag=tf.exp(t[..., 1])))
        self.action_proba = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(t[1])))

    def call(self, inputs) -> Tuple[Tensor, Tensor]:
        """
        Used for forward pass through head network.

        :param inputs: middleware state representation,
            of shape (batch_size, in_channels) or
            of shape (batch_size, time_step, in_channels).
        :return: batch of probabilities for each action,
            of shape (batch_size, action_mean) or
            of shape (batch_size, time_step, action_mean).
        """
        policy_means = self.policy_means_layer(inputs)
        log_stds = self.policy_log_std_layer#(inputs)

        ########
        a_prob = self.action_proba([policy_means, log_stds])
        #policy_means = a_prob.mean()
        #policy_std = a_prob.stddev()
        ########
        return a_prob
        #return policy_means, policy_std




class PPOHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_type_idx: int=0,
                 loss_weight: float=1.,
                 is_local: bool=True,
                 activation_function: str='tanh',
                 dense_layer: None=None) -> None:
        """
        Head block for Proximal Policy Optimization, to calculate probabilities for each action given middleware
        representation of the environment state.

        :param agent_parameters: containing algorithm parameters such as clip_likelihood_ratio_using_epsilon
            and beta_entropy.
        :param spaces: containing action spaces used for defining size of network output.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super().__init__(agent_parameters, spaces, network_name, head_type_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.return_type = ActionProbabilities

        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.beta = agent_parameters.algorithm.beta_entropy
        self.use_kl_regularization = agent_parameters.algorithm.use_kl_regularization
        if self.use_kl_regularization:
            self.initial_kl_coefficient = agent_parameters.algorithm.initial_kl_coefficient
            self.kl_cutoff = 2 * agent_parameters.algorithm.target_kl_divergence
            self.high_kl_penalty_coefficient = agent_parameters.algorithm.high_kl_penalty_coefficient
        else:
            self.initial_kl_coefficient, self.kl_cutoff, self.high_kl_penalty_coefficient = (None, None, None)
        self._loss = []

        if isinstance(self.spaces.action, BoxActionSpace):
            self.net = ContinuousPPOHead(num_actions=self.spaces.action.shape[0])
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for PPO.")

    def call(self, inputs):
        """
        :param inputs: middleware embedding
        :return: policy parameters/probabilities
        """
        return self.net(inputs)


    @property
    def kl_divergence(self):
        return self.head_type_idx, LOSS_OUT_TYPE_KL

    @property
    def entropy(self):
        return self.head_type_idx, LOSS_OUT_TYPE_ENTROPY

    @property
    def likelihood_ratio(self):
        return self.head_type_idx, LOSS_OUT_TYPE_LIKELIHOOD_RATIO

    @property
    def clipped_likelihood_ratio(self):
        return self.head_type_idx, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO

    def assign_kl_coefficient(self, kl_coefficient: float) -> None:
        self._loss[0]