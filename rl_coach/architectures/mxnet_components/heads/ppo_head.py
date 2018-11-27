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
from types import ModuleType

import math
import mxnet as mx
from mxnet.gluon import nn
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace
from rl_coach.utils import eps
from rl_coach.architectures.mxnet_components.heads.head import Head, HeadLoss, LossInputSchema,\
    NormalizedRSSInitializer
from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS, LOSS_OUT_TYPE_REGULARIZATION
from rl_coach.architectures.mxnet_components.utils import hybrid_clip, broadcast_like


LOSS_OUT_TYPE_KL = 'kl_divergence'
LOSS_OUT_TYPE_ENTROPY = 'entropy'
LOSS_OUT_TYPE_LIKELIHOOD_RATIO = 'likelihood_ratio'
LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO = 'clipped_likelihood_ratio'

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


class MultivariateNormalDist:
    def __init__(self,
                 num_var: int,
                 mean: nd_sym_type,
                 sigma: nd_sym_type,
                 F: ModuleType=mx.nd) -> None:
        """
        Distribution object for Multivariate Normal. Works with batches. 
        Optionally works with batches and time steps, but be consistent in usage: i.e. if using time_step,
        mean, sigma and data for log_prob must all include a time_step dimension.

        :param num_var: number of variables in distribution
        :param mean: mean for each variable,
            of shape (num_var) or
            of shape (batch_size, num_var) or
            of shape (batch_size, time_step, num_var).
        :param sigma: covariance matrix,
            of shape (num_var, num_var) or
            of shape (batch_size, num_var, num_var) or
            of shape (batch_size, time_step, num_var, num_var).
        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        """
        self.num_var = num_var
        self.mean = mean
        self.sigma = sigma
        self.F = F

    def inverse_using_cholesky(self, matrix: nd_sym_type) -> nd_sym_type:
        """
        Calculate inverses for a batch of matrices using Cholesky decomposition method.

        :param matrix: matrix (or matrices) to invert,
            of shape (num_var, num_var) or
            of shape (batch_size, num_var, num_var) or
            of shape (batch_size, time_step, num_var, num_var).
        :return: inverted matrix (or matrices),
            of shape (num_var, num_var) or
            of shape (batch_size, num_var, num_var) or
            of shape (batch_size, time_step, num_var, num_var).
        """
        cholesky_factor = self.F.linalg.potrf(matrix)
        return self.F.linalg.potri(cholesky_factor)

    def log_det(self, matrix: nd_sym_type) -> nd_sym_type:
        """
        Calculate log of the determinant for a batch of matrices using Cholesky decomposition method.

        :param matrix: matrix (or matrices) to invert,
            of shape (num_var, num_var) or
            of shape (batch_size, num_var, num_var) or
            of shape (batch_size, time_step, num_var, num_var).
        :return: inverted matrix (or matrices),
            of shape (num_var, num_var) or
            of shape (batch_size, num_var, num_var) or
            of shape (batch_size, time_step, num_var, num_var).
        """
        cholesky_factor = self.F.linalg.potrf(matrix)
        return 2 * self.F.linalg.sumlogdiag(cholesky_factor)

    def log_prob(self, x: nd_sym_type) -> nd_sym_type:
        """
        Calculate the log probability of data given the current distribution.

        See http://www.notenoughthoughts.net/posts/normal-log-likelihood-gradient.html
        and https://discuss.mxnet.io/t/multivariate-gaussian-log-density-operator/1169/7

        :param x: input data,
            of shape (num_var) or
            of shape (batch_size, num_var) or
            of shape (batch_size, time_step, num_var).
        :return: log_probability,
            of shape (1) or
            of shape (batch_size) or
            of shape (batch_size, time_step).
        """
        a = (self.num_var / 2) * math.log(2 * math.pi)
        log_det_sigma = self.log_det(self.sigma)
        b = (1 / 2) * log_det_sigma
        sigma_inv = self.inverse_using_cholesky(self.sigma)
        # deviation from mean, and dev_t is equivalent to transpose on last two dims.
        dev = (x - self.mean).expand_dims(-1)
        dev_t = (x - self.mean).expand_dims(-2)

        # since batch_dot only works with ndarrays with ndim of 3,
        # and we could have ndarrays with ndim of 4,
        # we flatten batch_size and time_step into single dim.
        dev_flat = dev.reshape(shape=(-1, 0, 0), reverse=1)
        sigma_inv_flat = sigma_inv.reshape(shape=(-1, 0, 0), reverse=1)
        dev_t_flat = dev_t.reshape(shape=(-1, 0, 0), reverse=1)
        c = (1 / 2) * self.F.batch_dot(self.F.batch_dot(dev_t_flat, sigma_inv_flat), dev_flat)
        # and now reshape back to (batch_size, time_step) if required.
        c = c.reshape_like(b)

        log_likelihood = -a - b - c
        return log_likelihood

    def entropy(self) -> nd_sym_type:
        """
        Calculate entropy of current distribution.

        See http://www.nowozin.net/sebastian/blog/the-entropy-of-a-normal-distribution.html
        :return: entropy,
            of shape (1) or
            of shape (batch_size) or
            of shape (batch_size, time_step).
        """
        # todo: check if differential entropy is correct
        log_det_sigma = self.log_det(self.sigma)
        return (self.num_var / 2) + ((self.num_var / 2) * math.log(2 * math.pi)) + ((1 / 2) * log_det_sigma)

    def kl_div(self, alt_dist) -> nd_sym_type:
        """
        Calculated KL-Divergence with another MultivariateNormalDist distribution
        See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        Specifically https://wikimedia.org/api/rest_v1/media/math/render/svg/a3bf3b4917bd1fcb8be48d6d6139e2e387bdc7d3

        :param alt_dist: alternative distribution used for kl divergence calculation
        :type alt_dist: MultivariateNormalDist
        :return: KL-Divergence, of shape (1,)
        """
        sigma_a_inv = self.F.linalg.potri(self.F.linalg.potrf(self.sigma))
        sigma_b_inv = self.F.linalg.potri(self.F.linalg.potrf(alt_dist.sigma))
        term1a = mx.nd.batch_dot(sigma_b_inv, self.sigma)
        # sum of diagonal for batch of matrices
        term1 = (broadcast_like(self.F, self.F.eye(self.num_var), term1a) * term1a).sum(axis=-1).sum(axis=-1)
        mean_diff = (alt_dist.mean - self.mean).expand_dims(-1)
        mean_diff_t = (alt_dist.mean - self.mean).expand_dims(-2)
        term2 = self.F.batch_dot(self.F.batch_dot(mean_diff_t, sigma_b_inv), mean_diff).reshape_like(term1)
        term3 = (2 * self.F.linalg.sumlogdiag(self.F.linalg.potrf(alt_dist.sigma))) -\
                (2 * self.F.linalg.sumlogdiag(self.F.linalg.potrf(self.sigma)))
        return 0.5 * (term1 + term2 - self.num_var + term3)


class CategoricalDist:
    def __init__(self, n_classes: int, probs: nd_sym_type, F: ModuleType=mx.nd) -> None:
        """
        Distribution object for Categorical data.
        Optionally works with batches and time steps, but be consistent in usage: i.e. if using time_step,
        mean, sigma and data for log_prob must all include a time_step dimension.

        :param n_classes: number of classes in distribution
        :param probs: probabilities for each class,
            of shape (n_classes),
            of shape (batch_size, n_classes) or
            of shape (batch_size, time_step, n_classes)
        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        """
        self.n_classes = n_classes
        self.probs = probs
        self.F = F


    def log_prob(self, actions: nd_sym_type) -> nd_sym_type:
        """
        Calculate the log probability of data given the current distribution.

        :param actions: actions, with int8 data type,
            of shape (1) if probs was (n_classes),
            of shape (batch_size) if probs was (batch_size, n_classes) and
            of shape (batch_size, time_step) if probs was (batch_size, time_step, n_classes)
        :return: log_probability,
            of shape (1) if probs was (n_classes),
            of shape (batch_size) if probs was (batch_size, n_classes) and
            of shape (batch_size, time_step) if probs was (batch_size, time_step, n_classes)
        """
        action_mask = actions.one_hot(depth=self.n_classes)
        action_probs = (self.probs * action_mask).sum(axis=-1)
        return action_probs.log()

    def entropy(self) -> nd_sym_type:
        """
        Calculate entropy of current distribution.

        :return: entropy,
            of shape (1) if probs was (n_classes),
            of shape (batch_size) if probs was (batch_size, n_classes) and
            of shape (batch_size, time_step) if probs was (batch_size, time_step, n_classes)
        """
        # todo: look into numerical stability
        return -(self.probs.log()*self.probs).sum(axis=-1)

    def kl_div(self, alt_dist) -> nd_sym_type:
        """
        Calculated KL-Divergence with another Categorical distribution

        :param alt_dist: alternative distribution used for kl divergence calculation
        :type alt_dist: CategoricalDist
        :return: KL-Divergence
        """
        logits_a = self.probs.clip(a_min=eps, a_max=1 - eps).log()
        logits_b = alt_dist.probs.clip(a_min=eps, a_max=1 - eps).log()
        t = self.probs * (logits_a - logits_b)
        t = self.F.where(condition=(alt_dist.probs == 0), x=self.F.ones_like(alt_dist.probs) * math.inf, y=t)
        t = self.F.where(condition=(self.probs == 0), x=self.F.zeros_like(self.probs), y=t)
        return t.sum(axis=-1)


class DiscretePPOHead(nn.HybridBlock):
    def __init__(self, num_actions: int) -> None:
        """
        Head block for Discrete Proximal Policy Optimization, to calculate probabilities for each action given
        middleware representation of the environment state.

        :param num_actions: number of actions in action space.
        """
        super(DiscretePPOHead, self).__init__()
        with self.name_scope():
            self.dense = nn.Dense(units=num_actions, flatten=False,
                                  weight_initializer=NormalizedRSSInitializer(0.01))

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type) -> nd_sym_type:
        """
        Used for forward pass through head network.

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param x: middleware state representation,
            of shape (batch_size, in_channels) or
            of shape (batch_size, time_step, in_channels).
        :return: batch of probabilities for each action,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        """
        policy_values = self.dense(x)
        policy_probs = F.softmax(policy_values)
        return policy_probs


class ContinuousPPOHead(nn.HybridBlock):
    def __init__(self, num_actions: int) -> None:
        """
        Head block for Continuous Proximal Policy Optimization, to calculate probabilities for each action given
        middleware representation of the environment state.

        :param num_actions: number of actions in action space.
        """
        super(ContinuousPPOHead, self).__init__()
        with self.name_scope():
            self.dense = nn.Dense(units=num_actions, flatten=False,
                                  weight_initializer=NormalizedRSSInitializer(0.01))
            # all samples (across batch, and time step) share the same covariance, which is learnt,
            # but since we assume the action probability variables are independent,
            # only the diagonal entries of the covariance matrix are specified.
            self.log_std = self.params.get('log_std',
                                           shape=(num_actions,),
                                           init=mx.init.Zero(),
                                           allow_deferred_init=True)
        # todo: is_local?

    def hybrid_forward(self, F: ModuleType, x: nd_sym_type, log_std: nd_sym_type) -> Tuple[nd_sym_type, nd_sym_type]:
        """
        Used for forward pass through head network.

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param x: middleware state representation,
            of shape (batch_size, in_channels) or
            of shape (batch_size, time_step, in_channels).
        :return: batch of probabilities for each action,
            of shape (batch_size, action_mean) or
            of shape (batch_size, time_step, action_mean).
        """
        policy_means = self.dense(x)
        policy_std = broadcast_like(F, log_std.exp().expand_dims(0), policy_means)
        return policy_means, policy_std


class ClippedPPOLossDiscrete(HeadLoss):
    def __init__(self,
                 num_actions: int,
                 clip_likelihood_ratio_using_epsilon: float,
                 beta: float=0,
                 use_kl_regularization: bool=False,
                 initial_kl_coefficient: float=1,
                 kl_cutoff: float=0,
                 high_kl_penalty_coefficient: float=1,
                 weight: float=1,
                 batch_axis: int=0) -> None:
        """
        Loss for discrete version of Clipped PPO.

        :param num_actions: number of actions in action space.
        :param clip_likelihood_ratio_using_epsilon: epsilon to use for likelihood ratio clipping.
        :param beta: loss coefficient applied to entropy
        :param use_kl_regularization: option to add kl divergence loss
        :param initial_kl_coefficient: loss coefficient applied kl divergence loss (also see high_kl_penalty_coefficient).
        :param kl_cutoff: threshold for using high_kl_penalty_coefficient
        :param high_kl_penalty_coefficient: loss coefficient applied to kv divergence above kl_cutoff
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super(ClippedPPOLossDiscrete, self).__init__(weight=weight, batch_axis=batch_axis)
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
            head_outputs=['new_policy_probs'],
            agent_inputs=['actions', 'old_policy_probs', 'clip_param_rescaler'],
            targets=['advantages']
        )

    def loss_forward(self,
                     F: ModuleType,
                     new_policy_probs: nd_sym_type,
                     actions: nd_sym_type,
                     old_policy_probs: nd_sym_type,
                     clip_param_rescaler: nd_sym_type,
                     advantages: nd_sym_type,
                     kl_coefficient: nd_sym_type) -> List[Tuple[nd_sym_type, str]]:
        """
        Used for forward pass through loss computations.
        Works with batches of data, and optionally time_steps, but be consistent in usage: i.e. if using time_step,
        new_policy_probs, old_policy_probs, actions and advantages all must include a time_step dimension.

        NOTE: order of input arguments MUST NOT CHANGE because it matches the order
        parameters are passed in ppo_agent:train_network()

        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param new_policy_probs: action probabilities predicted by DiscretePPOHead network,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param old_policy_probs: action probabilities for previous policy,
            of shape (batch_size, num_actions) or
            of shape (batch_size, time_step, num_actions).
        :param actions: true actions taken during rollout,
            of shape (batch_size) or
            of shape (batch_size, time_step).
        :param clip_param_rescaler: scales epsilon to use for likelihood ratio clipping.
        :param advantages: change in state value after taking action (a.k.a advantage)
            of shape (batch_size) or
            of shape (batch_size, time_step).
        :param kl_coefficient: loss coefficient applied kl divergence loss (also see high_kl_penalty_coefficient).
        :return: loss, of shape (batch_size).
        """

        old_policy_dist = CategoricalDist(self.num_actions, old_policy_probs, F=F)
        action_probs_wrt_old_policy = old_policy_dist.log_prob(actions)

        new_policy_dist = CategoricalDist(self.num_actions, new_policy_probs, F=F)
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


class ClippedPPOLossContinuous(HeadLoss):
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
                     F: ModuleType,
                     new_policy_means: nd_sym_type,
                     new_policy_stds: nd_sym_type,
                     actions: nd_sym_type,
                     old_policy_means: nd_sym_type,
                     old_policy_stds: nd_sym_type,
                     clip_param_rescaler: nd_sym_type,
                     advantages: nd_sym_type,
                     kl_coefficient: nd_sym_type) -> List[Tuple[nd_sym_type, str]]:
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

        if isinstance(self.spaces.action, DiscreteActionSpace):
            self.net = DiscretePPOHead(num_actions=len(self.spaces.action.actions))
        elif isinstance(self.spaces.action, BoxActionSpace):
            self.net = ContinuousPPOHead(num_actions=self.spaces.action.shape[0])
        else:
            raise ValueError("Only discrete or continuous action spaces are supported for PPO.")

    def hybrid_forward(self,
                       F: ModuleType,
                       x: nd_sym_type) -> nd_sym_type:
        """
        :param (mx.nd or mx.sym) F: backend api (mx.sym if block has been hybridized).
        :param x: middleware embedding
        :return: policy parameters/probabilities
        """
        return self.net(x)

    def loss(self) -> mx.gluon.loss.Loss:
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
        # set a property so can assign_kl_coefficient in future,
        # make a list, otherwise it would be added as a child of Head Block (due to type check)
        self._loss = [loss]
        return loss

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
        self._loss[0].kl_coefficient.set_data(mx.nd.array((kl_coefficient,)))