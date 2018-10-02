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

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head, normalized_columns_initializer, HeadParameters
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.exploration_policies.continuous_entropy import ContinuousEntropyParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, CompoundActionSpace
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps, indent_string


class PolicyHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='policy_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=Dense):
        super().__init__(parameterized_class=PolicyHead, activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)



class PolicyHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='tanh',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'policy_values_head'
        self.return_type = ActionProbabilities
        self.beta = None
        self.action_penalty = None

        self.exploration_policy = agent_parameters.exploration

        # a scalar weight that penalizes low entropy values to encourage exploration
        if hasattr(agent_parameters.algorithm, 'beta_entropy'):
            # we set the beta value as a tf variable so it can be updated later if needed
            self.beta = tf.Variable(float(agent_parameters.algorithm.beta_entropy),
                                    trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.beta_placeholder = tf.placeholder('float')
            self.set_beta = tf.assign(self.beta, self.beta_placeholder)

        # a scalar weight that penalizes high activation values (before the activation function) for the final layer
        if hasattr(agent_parameters.algorithm, 'action_penalty'):
            self.action_penalty = agent_parameters.algorithm.action_penalty

    def _build_module(self, input_layer):
        self.actions = []
        self.input = self.actions
        self.policy_distributions = []
        self.output = []

        action_spaces = [self.spaces.action]
        if isinstance(self.spaces.action, CompoundActionSpace):
            action_spaces = self.spaces.action.sub_action_spaces

        # create a compound action network
        for action_space_idx, action_space in enumerate(action_spaces):
            with tf.variable_scope("sub_action_{}".format(action_space_idx)):
                if isinstance(action_space, DiscreteActionSpace):
                    # create a discrete action network (softmax probabilities output)
                    self._build_discrete_net(input_layer, action_space)
                elif isinstance(action_space, BoxActionSpace):
                    # create a continuous action network (bounded mean and stdev outputs)
                    self._build_continuous_net(input_layer, action_space)

        if self.is_local:
            # add entropy regularization
            if self.beta:
                self.entropy = tf.add_n([tf.reduce_mean(dist.entropy()) for dist in self.policy_distributions])
                self.regularizations += [-tf.multiply(self.beta, self.entropy, name='entropy_regularization')]

            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizations)

            # calculate loss
            self.action_log_probs_wrt_policy = \
                tf.add_n([dist.log_prob(action) for dist, action in zip(self.policy_distributions, self.actions)])
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.target = self.advantages
            self.loss = -tf.reduce_mean(self.action_log_probs_wrt_policy * self.advantages)
            tf.losses.add_loss(self.loss_weight[0] * self.loss)

    def _build_discrete_net(self, input_layer, action_space):
        num_actions = len(action_space.actions)
        self.actions.append(tf.placeholder(tf.int32, [None], name="actions"))

        policy_values = self.dense_layer(num_actions)(input_layer, name='fc')
        self.policy_probs = tf.nn.softmax(policy_values, name="policy")

        # define the distributions for the policy and the old policy
        # (the + eps is to prevent probability 0 which will cause the log later on to be -inf)
        policy_distribution = tf.contrib.distributions.Categorical(probs=(self.policy_probs + eps))
        self.policy_distributions.append(policy_distribution)
        self.output.append(self.policy_probs)

    def _build_continuous_net(self, input_layer, action_space):
        num_actions = action_space.shape
        self.actions.append(tf.placeholder(tf.float32, [None, num_actions], name="actions"))

        # output activation function
        if np.all(action_space.max_abs_range < np.inf):
            # bounded actions
            self.output_scale = action_space.max_abs_range
            self.continuous_output_activation = self.activation_function
        else:
            # unbounded actions
            self.output_scale = 1
            self.continuous_output_activation = None

        # mean
        pre_activation_policy_values_mean = self.dense_layer(num_actions)(input_layer, name='fc_mean')
        policy_values_mean = self.continuous_output_activation(pre_activation_policy_values_mean)
        self.policy_mean = tf.multiply(policy_values_mean, self.output_scale, name='output_mean')

        self.output.append(self.policy_mean)

        # standard deviation
        if isinstance(self.exploration_policy, ContinuousEntropyParameters):
            # the stdev is an output of the network and uses a softplus activation as defined in A3C
            policy_values_std = self.dense_layer(num_actions)(input_layer,
                                                              kernel_initializer=normalized_columns_initializer(0.01),
                                                              name='fc_std')
            self.policy_std = tf.nn.softplus(policy_values_std, name='output_variance') + eps

            self.output.append(self.policy_std)
        else:
            # the stdev is an externally given value
            # Warning: we need to explicitly put this variable in the local variables collections, since defining
            # it as not trainable puts it for some reason in the global variables collections. If this is not done,
            # the variable won't be initialized and when working with multiple workers they will get stuck.
            self.policy_std = tf.Variable(np.ones(num_actions), dtype='float32', trainable=False,
                                          name='policy_stdev', collections=[tf.GraphKeys.LOCAL_VARIABLES])

            # assign op for the policy std
            self.policy_std_placeholder = tf.placeholder('float32', (num_actions,))
            self.assign_policy_std = tf.assign(self.policy_std, self.policy_std_placeholder)

        # define the distributions for the policy and the old policy
        policy_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean, self.policy_std)
        self.policy_distributions.append(policy_distribution)

        if self.is_local:
            # add a squared penalty on the squared pre-activation features of the action
            if self.action_penalty and self.action_penalty != 0:
                self.regularizations += [
                    self.action_penalty * tf.reduce_mean(tf.square(pre_activation_policy_values_mean))]

    def __str__(self):
        action_spaces = [self.spaces.action]
        if isinstance(self.spaces.action, CompoundActionSpace):
            action_spaces = self.spaces.action.sub_action_spaces

        result = []
        for action_space_idx, action_space in enumerate(action_spaces):
            action_head_mean_result = []
            if isinstance(action_space, DiscreteActionSpace):
                # create a discrete action network (softmax probabilities output)
                action_head_mean_result.append("Dense (num outputs = {})".format(len(action_space.actions)))
                action_head_mean_result.append("Softmax")
            elif isinstance(action_space, BoxActionSpace):
                # create a continuous action network (bounded mean and stdev outputs)
                action_head_mean_result.append("Dense (num outputs = {})".format(action_space.shape))
                if np.all(action_space.max_abs_range < np.inf):
                    # bounded actions
                    action_head_mean_result.append("Activation (type = {})".format(self.activation_function.__name__))
                    action_head_mean_result.append("Multiply (factor = {})".format(action_space.max_abs_range))

            action_head_stdev_result = []
            if isinstance(self.exploration_policy, ContinuousEntropyParameters):
                action_head_stdev_result.append("Dense (num outputs = {})".format(action_space.shape))
                action_head_stdev_result.append("Softplus")

            action_head_result = []
            if action_head_stdev_result:
                action_head_result.append("Mean Stream")
                action_head_result.append(indent_string('\n'.join(action_head_mean_result)))
                action_head_result.append("Stdev Stream")
                action_head_result.append(indent_string('\n'.join(action_head_stdev_result)))
            else:
                action_head_result.append('\n'.join(action_head_mean_result))

            if len(action_spaces) > 1:
                result.append("Action head {}".format(action_space_idx))
                result.append(indent_string('\n'.join(action_head_result)))
            else:
                result.append('\n'.join(action_head_result))

        return '\n'.join(result)
