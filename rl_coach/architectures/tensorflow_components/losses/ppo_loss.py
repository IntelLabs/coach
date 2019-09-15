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
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError
from typing import List, Tuple

from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema, LOSS_OUT_TYPE_LOSS
from tensorflow import Tensor

class ClippedPPOLossContinuous(Loss):
    def __init__(self,
                 num_actions: int,
                 clip_likelihood_ratio_using_epsilon: float,
                 beta: float=0,
                 use_kl_regularization: bool=False,
                 initial_kl_coefficient: float=1,
                 kl_cutoff: float=0,
                 high_kl_penalty_coefficient: float=1,
                 weight: float=1,
                 batch_axis: int=0):
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
        super(ClippedPPOLossContinuous, self).__init__(weight=weight, batch_axis=batch_axis)
        self.weight = weight
        self.num_actions = num_actions
        self.clip_likelihood_ratio_using_epsilon = clip_likelihood_ratio_using_epsilon
        self.beta = beta
        self.use_kl_regularization = use_kl_regularization
        self.initial_kl_coefficient = initial_kl_coefficient if self.use_kl_regularization else 0.0
        self.kl_coefficient = self.params.get('kl_coefficient',
                                              shape=(1,),
                                              init=mx.init.Constant([initial_kl_coefficient,]),
                                              differentiable=False)
        self.kl_cutoff = kl_cutoff
        self.high_kl_penalty_coefficient = high_kl_penalty_coefficient

    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            head_outputs=['new_policy_means','new_policy_stds'],
            agent_inputs=['actions', 'old_policy_means', 'old_policy_stds', 'clip_param_rescaler'],
            targets=['advantages']
        )

    def loss_forward(self,
                     new_policy_means,
                     new_policy_stds,
                     actions,
                     old_policy_means,
                     old_policy_stds,
                     clip_param_rescaler,
                     advantages,
                     kl_coefficient) -> List[Tuple[Tensor, str]]:
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

        def diagonal_covariance(stds, size):
            vars = stds ** 2
            # sets diagonal in (batch size and time step) covariance matrices
            vars_tiled = vars.expand_dims(2).tile((1, 1, size))
            covars = F.broadcast_mul(vars_tiled, F.eye(size))
            return covars

        old_covar = diagonal_covariance(stds=old_policy_stds, size=self.num_actions)
        old_policy_dist = MultivariateNormalDist(self.num_actions, old_policy_means, old_covar, F=F)
        action_probs_wrt_old_policy = old_policy_dist.log_prob(actions)

        new_covar = diagonal_covariance(stds=new_policy_stds, size=self.num_actions)
        new_policy_dist = MultivariateNormalDist(self.num_actions, new_policy_means, new_covar, F=F)
        action_probs_wrt_new_policy = new_policy_dist.log_prob(actions)

        entropy_loss = - self.beta * new_policy_dist.entropy().mean()

        if self.use_kl_regularization:
            kl_div = old_policy_dist.kl_div(new_policy_dist).mean()
            weighted_kl_div = kl_coefficient * kl_div
            high_kl_div = F.stack(F.zeros_like(kl_div), kl_div - self.kl_cutoff).max().square()
            weighted_high_kl_div = self.high_kl_penalty_coefficient * high_kl_div
            kl_div_loss = weighted_kl_div + weighted_high_kl_div
        else:
            kl_div_loss = F.zeros(shape=(1,))

        # working with log probs, so minus first, then exponential (same as division)
        likelihood_ratio = (action_probs_wrt_new_policy - action_probs_wrt_old_policy).exp()

        if self.clip_likelihood_ratio_using_epsilon is not None:
            # clipping of likelihood ratio
            min_value = 1 - self.clip_likelihood_ratio_using_epsilon * clip_param_rescaler
            max_value = 1 + self.clip_likelihood_ratio_using_epsilon * clip_param_rescaler

            # can't use F.clip (with variable clipping bounds), hence custom implementation
            clipped_likelihood_ratio = hybrid_clip(F, likelihood_ratio, clip_lower=min_value, clip_upper=max_value)

            # lower bound of original, and clipped versions or each scaled advantage
            # element-wise min between the two ndarrays
            unclipped_scaled_advantages = likelihood_ratio * advantages
            clipped_scaled_advantages = clipped_likelihood_ratio * advantages
            scaled_advantages = F.stack(unclipped_scaled_advantages, clipped_scaled_advantages).min(axis=0)
        else:
            scaled_advantages = likelihood_ratio * advantages
            clipped_likelihood_ratio = F.zeros_like(likelihood_ratio)

        # for each batch, calculate expectation of scaled_advantages across time steps,
        # but want code to work with data without time step too, so reshape to add timestep if doesn't exist.
        scaled_advantages_w_time = scaled_advantages.reshape(shape=(0, -1))
        expected_scaled_advantages = scaled_advantages_w_time.mean(axis=1)
        # want to maximize expected_scaled_advantages, add minus so can minimize.
        surrogate_loss = (-expected_scaled_advantages * self.weight).mean()

        return [
            (surrogate_loss, LOSS_OUT_TYPE_LOSS),
            (entropy_loss + kl_div_loss, LOSS_OUT_TYPE_REGULARIZATION),
            (kl_div_loss, LOSS_OUT_TYPE_KL),
            (entropy_loss, LOSS_OUT_TYPE_ENTROPY),
            (likelihood_ratio, LOSS_OUT_TYPE_LIKELIHOOD_RATIO),
            (clipped_likelihood_ratio, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO)
        ]


class PPOLoss(keras.losses.Loss):

    def __init__(self, network_name,
                 head_idx: int = 0,
                 loss_type: Loss = MeanSquaredError,
                 loss_weight=1.0,
                 **kwargs):
        """
        Loss for Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)


    def call(self, prediction, target):
        """
        Used for forward pass through loss computations.

        :param prediction: state values predicted by VHead network, of shape (batch_size).
        :param target: actual state values, of shape (batch_size).
        :return: loss, of shape (batch_size).
        """
        # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
        #  This way, Keras can apply class weights or sample weights when requested.
        loss = tf.reduce_mean(self.loss_fn(prediction, target))
        return loss


        """
        Specifies loss block to be used for this policy head.

        :return: loss block (can be called as function) for action probabilities returned by this policy network.
        """
        if isinstance(self.spaces.action, DiscreteActionSpace):
            loss = ClippedPPOLossDiscrete(len(self.spaces.action.actions),
                                          self.clip_likelihood_ratio_using_epsilon,
                                          self.beta,
                                          self.use_kl_regularization, self.initial_kl_coefficient,
                                          self.kl_cutoff, self.high_kl_penalty_coefficient,
                                          self.loss_weight)
        elif isinstance(self.spaces.action, BoxActionSpace):
            loss = ClippedPPOLossContinuous(self.spaces.action.shape[0],
                                            self.clip_likelihood_ratio_using_epsilon,
                                            self.beta,
                                            self.use_kl_regularization, self.initial_kl_coefficient,
                                            self.kl_cutoff, self.high_kl_penalty_coefficient,
                                            self.loss_weight)
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for PPO.")
        loss.initialize()

        self._loss = [loss]
        return loss