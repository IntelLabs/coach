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

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError
from typing import List, Tuple

from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema,\
    LOSS_OUT_TYPE_LOSS, LOSS_OUT_TYPE_REGULARIZATION
from tensorflow import Tensor

import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability import edward2 as ed

LOSS_OUT_TYPE_KL = 'kl_divergence'
LOSS_OUT_TYPE_ENTROPY = 'entropy'
LOSS_OUT_TYPE_LIKELIHOOD_RATIO = 'likelihood_ratio'
LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO = 'clipped_likelihood_ratio'
from rl_coach.utils import eps

from tensorflow.keras.layers import Dense, Input, Lambda



class PPOLoss(HeadLoss):
    def __init__(self,
                 network_name,
                 agent_parameters,
                 num_actions,
                 head_idx,
                 loss_type,
                 loss_weight):

        """
        Loss for continuous version of Clipped PPO.

        :param num_actions: number of actions in action space.
        :param clip_likelihood_ratio_using_epsilon: epsilon to use for likelihood ratio clipping.
        :param beta: loss coefficient applied to entropy
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        :param use_kl_regularization: option to add kl divergence loss
        :param initial_kl_coefficient: initial loss coefficient applied kl divergence loss (also see high_kl_penalty_coefficient).
        :param kl_cutoff: threshold for using high_kl_penalty_coefficient
        :param high_kl_penalty_coefficient: loss coefficient applied to kv divergence above kl_cutoff
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super(PPOLoss, self).__init__()
        self.weight = loss_weight
        self.num_actions = num_actions
        self.clip_likelihood_ratio_using_epsilon = agent_parameters.algorithm.clip_likelihood_ratio_using_epsilon
        self.beta = agent_parameters.algorithm.beta_entropy
        self.use_kl_regularization = agent_parameters.algorithm.use_kl_regularization

        if self.use_kl_regularization:
            self.initial_kl_coefficient = agent_parameters.algorithm.initial_kl_coefficient
            self.kl_cutoff = 2 * agent_parameters.algorithm.target_kl_divergence
            self.high_kl_penalty_coefficient = agent_parameters.algorithm.high_kl_penalty_coefficient
        else:
            self.initial_kl_coefficient, self.kl_cutoff, self.high_kl_penalty_coefficient = (0.0, None, None)



    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            #head_outputs=['new_policy_means', 'new_policy_stds'],
            head_outputs=['new_policy_distribution'],
            agent_inputs=['actions', 'old_policy_means', 'old_policy_stds', 'clip_param_rescaler'],
            targets=['advantages']
        )

    # def loss_forward(self,
    #                  new_policy_means,
    #                  new_policy_stds,
    #                  actions,
    #                  old_policy_means,
    #                  old_policy_stds,
    #                  clip_param_rescaler,
    #                  advantages) -> List[Tuple[Tensor, str]]:

    def loss_forward(self,
                     new_policy_distribution,
                     actions,
                     old_policy_means,
                     old_policy_stds,
                     clip_param_rescaler,
                     advantages) -> List[Tuple[Tensor, str]]:

        """
        Used for forward pass through loss computations.
        Works with batches of data, and optionally time_steps, but be consistent in usage: i.e. if using time_step,
        new_policy_means, old_policy_means, actions and advantages all must include a time_step dimension.

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param new_policy_means: action means predicted by MultivariateNormalDist network,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param new_policy_stds: action standard deviation returned by head,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param actions: true actions taken during rollout,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param old_policy_means: action means for previous policy,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param old_policy_stds: action standard deviation returned by head previously,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param clip_param_rescaler: scales epsilon to use for likelihood ratio clipping.
        :param advantages: change in state value after taking action (a.k.a advantage)
            of shape (batch_size,) or
            of shape (batch_size, time_step).
        :param kl_coefficient: loss coefficient applied kl divergence loss (also see high_kl_penalty_coefficient).
        :return: loss, of shape (batch_size).
        """
        old_policy_dist = tfd.MultivariateNormalDiag(loc=old_policy_means, scale_diag=old_policy_stds)# + eps)
        action_probs_wrt_old_policy = old_policy_dist.log_prob(actions)

        #new_policy_distribution = tfd.MultivariateNormalDiag(loc=new_policy_means, scale_diag=new_policy_stds)  # + eps)
        action_probs_wrt_new_policy = new_policy_distribution.log_prob(actions)

        entropy_loss = - self.beta * tf.reduce_mean(new_policy_distribution.entropy())

        assert self.use_kl_regularization == False # Not supported yet
        kl_div_loss = tf.constant(0, dtype=tf.float32)
        # working with log probs, so minus first, then exponential (same as division)
        likelihood_ratio = tf.exp(action_probs_wrt_new_policy - action_probs_wrt_old_policy)
        # Added when changed to functional
        # advantages = np.float32(advantages).reshape(likelihood_ratio.shape)

        if self.clip_likelihood_ratio_using_epsilon is not None:
            # clipping of likelihood ratio
            min_value = 1 - self.clip_likelihood_ratio_using_epsilon * clip_param_rescaler
            max_value = 1 + self.clip_likelihood_ratio_using_epsilon * clip_param_rescaler

            clipped_likelihood_ratio = tf.clip_by_value(likelihood_ratio, min_value, max_value)
            # lower bound of original, and clipped versions or each scaled advantage
            # element-wise min between the two arrays
            unclipped_scaled_advantages = likelihood_ratio * advantages
            clipped_scaled_advantages = clipped_likelihood_ratio * advantages
            scaled_advantages = tf.minimum(unclipped_scaled_advantages, clipped_scaled_advantages)

        else:
            scaled_advantages = likelihood_ratio * advantages
            clipped_likelihood_ratio = tf.zeros_like(likelihood_ratio)

        # for each batch, calculate expectation of scaled_advantages across time steps,
        # but want code to work with data without time step too, so reshape to add timestep if doesn't exist.
        expected_scaled_advantages = tf.reduce_mean(scaled_advantages)
        # want to maximize expected_scaled_advantages, add minus so can minimize.
        surrogate_loss = -expected_scaled_advantages * self.weight



        return [
            (surrogate_loss, LOSS_OUT_TYPE_LOSS),
            (entropy_loss + kl_div_loss, LOSS_OUT_TYPE_REGULARIZATION),
            (kl_div_loss, LOSS_OUT_TYPE_KL),
            (entropy_loss, LOSS_OUT_TYPE_ENTROPY),
            (likelihood_ratio, LOSS_OUT_TYPE_LIKELIHOOD_RATIO),
            (clipped_likelihood_ratio, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO)
        ]



        # dummy_loss = tf.reduce_mean(advantages*new_policy_distribution.log_prob(actions[1]))
        # dummy_loss = tf.reduce_mean(tf.sqrt(tf.square(new_policy_distribution.mean()- 5)))
        # print(new_policy_distribution.mean())
        #dummy_loss = tf.reduce_mean(advantages*new_policy_distribution.log_prob())
        #print(new_policy_distribution.mean())
        #dummy_loss = tf.constant(0, dtype=tf.float32)
        # return [
        #     (surrogate_loss, LOSS_OUT_TYPE_LOSS),
        #     (1e-10*(entropy_loss + kl_div_loss), LOSS_OUT_TYPE_REGULARIZATION),
        #     (1e-10*kl_div_loss, LOSS_OUT_TYPE_KL),
        #     (1e-10*entropy_loss, LOSS_OUT_TYPE_ENTROPY),

        #     (1e-10*likelihood_ratio, LOSS_OUT_TYPE_LIKELIHOOD_RATIO),
        #     (1e-10*clipped_likelihood_ratio, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO)
        # ]

# order shoud be y_tensor, model.output)

#def ppo_loss_f(targets, new_policy_rv):
def ppo_loss_f(advantages, old_means, old_stds, actions, rescalar, new_policy_rv):
    # actions = targets[0]
    # old_means = targets[1]
    # old_stds = targets[2]
    # advantages = targets[3]
    min_value = 1 - rescalar
    max_value = 1 + rescalar

    #x = Input([1])

    # actions_proba = tfp.layers.DistributionLambda(
    #     lambda t: tfd.MultivariateNormalDiag(
    #         loc=t[0], scale_diag=t[1]))([old_means, old_stds])
    #
    # model = keras.Model(name='continuous_ppo_head', inputs=inputs, outputs=actions_proba)

    old_policy = tfd.MultivariateNormalDiag(loc=old_means, scale_diag=old_stds + eps)

    # new_policy_log_prob = tfp.layers.DistributionLambda(
    #     lambda t: new_policy.log_prob(t))(actions)

    # new_policy_log_prob = old_policy.log_prob(actions)
    # old_policy_log_prob = new_policy_rv.log_prob(actions)
    old_policy_log_prob = old_policy.log_prob(actions)
    new_policy_log_prob = new_policy_rv.log_prob(actions)
    likelihood_ratio = tf.exp(new_policy_log_prob - old_policy_log_prob)

    clipped_likelihood_ratio = tf.clip_by_value(likelihood_ratio, min_value, max_value)
    clipped_scaled_advantages = clipped_likelihood_ratio * advantages
    unclipped_scaled_advantages = likelihood_ratio * advantages

    scaled_advantages = tf.minimum(unclipped_scaled_advantages, clipped_scaled_advantages)
    loss = -tf.reduce_mean(scaled_advantages)

    return loss


# def ppo_loss_f(new_policy_rv, actions, old_means, old_stds, advantages):
#
#     min_value = 0.8
#     max_value = 1.2
#     old_policy = tfd.MultivariateNormalDiag(loc=old_means, scale_diag=old_stds)
#
#     old_policy_log_prob = old_policy.log_prob(actions)
#     new_policy_log_prob = new_policy_rv.log_prob(actions)
#     likelihood_ratio = tf.exp(new_policy_log_prob - old_policy_log_prob)
#     clipped_likelihood_ratio = tf.clip_by_value(likelihood_ratio, min_value, max_value)
#     unclipped_scaled_advantages = likelihood_ratio * advantages
#     clipped_scaled_advantages = clipped_likelihood_ratio * advantages
#     scaled_advantages = tf.minimum(unclipped_scaled_advantages, clipped_scaled_advantages)
#     loss = -tf.reduce_mean(scaled_advantages)
#     return loss
#


# class PPOLoss(keras.losses.Loss):
#
#     def __init__(self, network_name,
#                  head_idx: int = 0,
#                  loss_type: Loss = MeanSquaredError,
#                  loss_weight=1.0,
#                  **kwargs):
#         """
#         Loss for Value Head.
#
#         :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
#         :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
#         :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
#         """
#         super().__init__(**kwargs)
#         self.loss_type = loss_type
#         self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)
#
#
#     def call(self, prediction, target):
#         """
#         Used for forward pass through loss computations.
#
#         :param prediction: state values predicted by VHead network, of shape (batch_size).
#         :param target: actual state values, of shape (batch_size).
#         :return: loss, of shape (batch_size).
#         """
#         # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
#         #  This way, Keras can apply class weights or sample weights when requested.
#         loss = tf.reduce_mean(self.loss_fn(prediction, target))
#         return loss
#
#
#         """
#         Specifies loss block to be used for this policy head.
#
#         :return: loss block (can be called as function) for action probabilities returned by this policy network.
#         """
#         if isinstance(self.spaces.action, DiscreteActionSpace):
#             loss = ClippedPPOLossDiscrete(len(self.spaces.action.actions),
#                                           self.clip_likelihood_ratio_using_epsilon,
#                                           self.beta,
#                                           self.use_kl_regularization, self.initial_kl_coefficient,
#                                           self.kl_cutoff, self.high_kl_penalty_coefficient,
#                                           self.loss_weight)
#         elif isinstance(self.spaces.action, BoxActionSpace):
#             loss = ClippedPPOLossContinuous(self.spaces.action.shape[0],
#                                             self.clip_likelihood_ratio_using_epsilon,
#                                             self.beta,
#                                             self.use_kl_regularization, self.initial_kl_coefficient,
#                                             self.kl_cutoff, self.high_kl_penalty_coefficient,
#                                             self.loss_weight)
#         else:
#             raise ValueError("Only discrete or continuous action spaces are supported for PPO.")
#         loss.initialize()
#
#         self._loss = [loss]
#         return loss