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
from tensorflow.keras.layers import Dense, Input, Lambda
#from tensorflow_probability.layers import DistributionLambda

import tensorflow_probability as tfp

tfd = tfp.distributions
import numpy as np
import tensorflow as tf

#from rl_coach.architectures.tensorflow_components.layers import Dense
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
        self.output_dim = num_actions
        # all samples (across batch, and time step) share the same covariance, which is learnt,
        # but since we assume the action probability variables are independent,
        # only the diagonal entries of the covariance matrix are specified.
        self.policy_means_layer = tf.keras.layers.Dense(units=num_actions)
        # self.policy_log_std_layer = tf.keras.layers.Dense(units=num_actions,
        #                                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=None),
        #                                                   bias_initializer='zeros')

        self.policy_log_std_layer = tf.Variable(tf.zeros((1, num_actions)), dtype='float32', name="policy_log_std")
        self.repeat_layer = tf.keras.layers.RepeatVector(64)

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
        log_stds = self.policy_log_std_layer


        if not inputs.shape[0]:
            log_stds = self.policy_log_std_layer#(inputs)
        #log_stds = tf.tile(self.policy_log_std_layer, [policy_means.shape[0], 1], name='log_std')
        else:
            #print('aa')
            log_stds = tf.tile(self.policy_log_std_layer, [inputs.shape[0], self.output_dim], name='log_std')


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
            #self.net = ContinuousPPOHead(num_actions=self.spaces.action.shape[0])
            head_input_dim = 64 # middleware output dim hard coded, because scheme is hard coded
            head_output_dim = self.spaces.action.shape[0]
            #self.net_f = self.continuous_ppo_head(head_input_dim, head_output_dim)
            self.net = self.continuous_ppo_head(head_input_dim, head_output_dim)
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for PPO.")

    def call(self, inputs):
        """
        :param inputs: middleware embedding
        :return: policy parameters/probabilities
        """
        # a = self.net(inputs)
        # b = self.net_f(inputs)
        # c = 1
        return self.net(inputs)


    def continuous_ppo_head(self, input_dim, output_dim):

        # def repeat_vector(args):
        #     layer_to_repeat = args[0]
        #     reference_layer = args[1]
        #     batch_size = tf.shape(reference_layer)[0]
        #     return tf.keras.layers.RepeatVector(batch_size)(layer_to_repeat)

        inputs = Input(shape=([input_dim]))
        #dummy_input = tf.ones((1, 1))
        policy_means = Dense(units=output_dim, name="policy_means")(inputs)
        policy_log_std = Bias()(inputs)#Dense(units=output_dim, name="policy_log_std")(inputs)
        #policy_log_std = tf.Variable(tf.zeros((1, output_dim)), dtype='float32', name="policy_log_std")
        #policy_log_std = Lambda(repeat_vector)([policy_log_std, policy_means])
        policy_stds = Lambda(lambda x: tf.exp(x))(policy_log_std)
        #policy_stds = Lambda(lambda x: tf.squeeze(x, 0))(policy_stds)
        actions_proba = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1]))([policy_means, policy_stds])

        model = keras.Model(name='continuous_ppo_head', inputs=inputs, outputs=actions_proba)
        #model.layers[-1].trainable_weights.append(policy_log_std)

        return model



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



class Bias(keras.layers.Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1,), initializer='zeros', dtype=tf.float32, name='log_std_var')
        super().build(input_shape)

    def call(self, x):
        temp = tf.reduce_mean(x, axis=-1, keepdims=True)
        return temp * 0 + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)