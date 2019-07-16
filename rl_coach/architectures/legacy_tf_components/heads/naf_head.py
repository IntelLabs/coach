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

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import BoxActionSpace
from rl_coach.spaces import SpacesDefinition


class NAFHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True,activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        if not isinstance(self.spaces.action, BoxActionSpace):
            raise ValueError("NAF works only for continuous action spaces (BoxActionSpace)")

        self.name = 'naf_q_values_head'
        self.num_actions = self.spaces.action.shape[0]
        self.output_scale = self.spaces.action.max_abs_range
        self.return_type = QActionStateValue
        if agent_parameters.network_wrappers[self.network_name].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

    def _build_module(self, input_layer):
        # NAF
        self.action = tf.placeholder(tf.float32, [None, self.num_actions], name="action")
        self.input = self.action

        # V Head
        self.V = self.dense_layer(1)(input_layer, name='V')

        # mu Head
        mu_unscaled = self.dense_layer(self.num_actions)(input_layer, activation=self.activation_function, name='mu_unscaled')
        self.mu = tf.multiply(mu_unscaled, self.output_scale, name='mu')

        # A Head
        # l_vector is a vector that includes a lower-triangular matrix values
        self.l_vector = self.dense_layer((self.num_actions * (self.num_actions + 1)) / 2)(input_layer, name='l_vector')

        # Convert l to a lower triangular matrix and exponentiate its diagonal

        i = 0
        columns = []
        for col in range(self.num_actions):
            start_row = col
            num_non_zero_elements = self.num_actions - start_row
            zeros_column_part = tf.zeros_like(self.l_vector[:, 0:start_row])
            diag_element = tf.expand_dims(tf.exp(self.l_vector[:, i]), 1)
            non_zeros_non_diag_column_part = self.l_vector[:, (i + 1):(i + num_non_zero_elements)]
            columns.append(tf.concat([zeros_column_part, diag_element, non_zeros_non_diag_column_part], axis=1))
            i += num_non_zero_elements
        self.L = tf.transpose(tf.stack(columns, axis=1), (0, 2, 1))

        # P = L*L^T
        self.P = tf.matmul(self.L, tf.transpose(self.L, (0, 2, 1)))

        # A = -1/2 * (u - mu)^T * P * (u - mu)
        action_diff = tf.expand_dims(self.action - self.mu, -1)
        a_matrix_form = -0.5 * tf.matmul(tf.transpose(action_diff, (0, 2, 1)), tf.matmul(self.P, action_diff))
        self.A = tf.reshape(a_matrix_form, [-1, 1])

        # Q Head
        self.Q = tf.add(self.V, self.A, name='Q')

        self.output = self.Q

    def __str__(self):
        result = [
            "State Value Stream - V",
            "\tDense (num outputs = 1)",
            "Action Advantage Stream - A",
            "\tDense (num outputs = {})".format((self.num_actions * (self.num_actions + 1)) / 2),
            "\tReshape to lower triangular matrix L (new size = {} x {})".format(self.num_actions, self.num_actions),
            "\tP = L*L^T",
            "\tA = -1/2 * (u - mu)^T * P * (u - mu)",
            "Action Stream - mu",
            "\tDense (num outputs = {})".format(self.num_actions),
            "\tActivation (type = {})".format(self.activation_function.__name__),
            "\tMultiply (factor = {})".format(self.output_scale),
            "State-Action Value Stream - Q",
            "\tAdd (V, A)"
        ]
        return '\n'.join(result)
