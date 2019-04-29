#
# Copyright (c) 2019 Intel Corporation
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

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class SACPolicyHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 squash: bool = True, dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'sac_policy_head'
        self.return_type = ActionProbabilities
        self.num_actions = self.spaces.action.shape     # continuous actions
        self.squash = squash        # squashing using tanh

    def _build_module(self, input_layer):
        self.given_raw_actions = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
        self.input = [self.given_raw_actions]
        self.output = []

        # build the network
        self._build_continuous_net(input_layer, self.spaces.action)

    def _squash_correction(self,actions):
        '''
        correct squash operation (in case of bounded actions) according to appendix C in the paper.
        NOTE : this correction assume the squash is done with tanh.
        :param actions: unbounded actions
        :return: the correction to be applied to the log_prob of the actions, assuming tanh squash
        '''
        if not self.squash:
            return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + eps), axis=1)

    def _build_continuous_net(self, input_layer, action_space):
        num_actions = action_space.shape[0]

        self.policy_mu_and_logsig = self.dense_layer(2*num_actions)(input_layer, name='policy_mu_logsig')
        self.policy_mean = tf.identity(self.policy_mu_and_logsig[..., :num_actions], name='policy_mean')
        self.policy_log_std = tf.clip_by_value(self.policy_mu_and_logsig[..., num_actions:],
                                               LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX,name='policy_log_std')

        self.output.append(self.policy_mean)        # output[0]
        self.output.append(self.policy_log_std)     # output[1]

        # define the distributions for the policy
        # Tensorflow's multivariate normal distribution supports reparameterization
        tfd = tf.contrib.distributions
        self.policy_distribution = tfd.MultivariateNormalDiag(loc=self.policy_mean,
                                                              scale_diag=tf.exp(self.policy_log_std))

        # define network outputs
        # note that tensorflow supports reparametrization.
        # i.e. policy_action_sample is a tensor through which gradients can flow
        self.raw_actions = self.policy_distribution.sample()

        if self.squash:
            self.actions = tf.tanh(self.raw_actions)
            # correct log_prob in case of squash (see appendix C in the paper)
            squash_correction = self._squash_correction(self.raw_actions)
        else:
            self.actions = self.raw_actions
            squash_correction = 0

        # policy_action_logprob is a tensor through which gradients can flow
        self.sampled_actions_logprob = self.policy_distribution.log_prob(self.raw_actions) - squash_correction
        self.sampled_actions_logprob_mean = tf.reduce_mean(self.sampled_actions_logprob)

        self.output.append(self.raw_actions)    # output[2] : sampled raw action (before squash)
        self.output.append(self.actions)        # output[3] : squashed (if needed) version of sampled raw_actions
        self.output.append(self.sampled_actions_logprob)   # output[4]: log prob of sampled action (squash corrected)
        self.output.append(self.sampled_actions_logprob_mean)    # output[5]: mean of log prob of sampled actions (squash corrected)

    def __str__(self):
        result = [
            "policy head:"
            "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = {0})".format(2*self.num_actions),
            "policy_mu = output[:num_actions], policy_std = output[num_actions:]"
        ]
        return '\n'.join(result)
