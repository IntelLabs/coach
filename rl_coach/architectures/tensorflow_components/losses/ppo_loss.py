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
import tensorflow_probability as tfp
from tensorflow import Tensor
from typing import List, Tuple, Dict
from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema,\
    LOSS_OUT_TYPE_LOSS, LOSS_OUT_TYPE_REGULARIZATION


tfd = tfp.distributions
LOSS_OUT_TYPE_KL = 'kl_divergence'
LOSS_OUT_TYPE_ENTROPY = 'entropy'
LOSS_OUT_TYPE_LIKELIHOOD_RATIO = 'likelihood_ratio'
LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO = 'clipped_likelihood_ratio'


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
        super(PPOLoss, self).__init__(name=network_name)
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
            head_outputs=['new_policy_distribution'],
            agent_inputs=['actions', 'old_policy_means', 'old_policy_stds', 'clip_param_rescaler'],
            targets=['advantages']
        )

    def loss_forward(self,
                     new_policy_distribution,
                     actions,
                     old_policy_means,
                     old_policy_stds,
                     clip_param_rescaler,
                     advantages):#-> Dict: #List[Tuple[Tensor, str]]:

        """
        Used for forward pass through loss computations.
        Works with batches of data, and optionally time_steps, but be consistent in usage: i.e. if using time_step,
        new_policy_means, old_policy_means, actions and advantages all must include a time_step dimension.
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


        return {
            LOSS_OUT_TYPE_LOSS: [surrogate_loss],
            LOSS_OUT_TYPE_REGULARIZATION: [(entropy_loss + kl_div_loss)],
            LOSS_OUT_TYPE_KL: kl_div_loss,
            LOSS_OUT_TYPE_ENTROPY: [entropy_loss],
            LOSS_OUT_TYPE_LIKELIHOOD_RATIO: [likelihood_ratio],
            LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO: [clipped_likelihood_ratio],
        }

        # return [
        #     (surrogate_loss, LOSS_OUT_TYPE_LOSS),
        #     (entropy_loss + kl_div_loss, LOSS_OUT_TYPE_REGULARIZATION),
        #     (kl_div_loss, LOSS_OUT_TYPE_KL),
        #     (entropy_loss, LOSS_OUT_TYPE_ENTROPY),
        #     (likelihood_ratio, LOSS_OUT_TYPE_LIKELIHOOD_RATIO),
        #     (clipped_likelihood_ratio, LOSS_OUT_TYPE_CLIPPED_LIKELIHOOD_RATIO)
        # ]



