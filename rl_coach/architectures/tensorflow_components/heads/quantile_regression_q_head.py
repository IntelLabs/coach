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

from rl_coach.architectures.tensorflow_components.architecture import Dense

from rl_coach.architectures.tensorflow_components.heads.head import Head, HeadParameters
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition


class QuantileRegressionQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='quantile_regression_q_head_params',
                 dense_layer=Dense):
        super().__init__(parameterized_class=QuantileRegressionQHead, activation_function=activation_function,
                         name=name, dense_layer=dense_layer)


class QuantileRegressionQHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'quantile_regression_dqn_head'
        self.num_actions = len(self.spaces.action.actions)
        self.num_atoms = agent_parameters.algorithm.atoms  # we use atom / quantile interchangeably
        self.huber_loss_interval = agent_parameters.algorithm.huber_loss_interval  # k
        self.return_type = QActionStateValue

    def _build_module(self, input_layer):
        self.actions = tf.placeholder(tf.int32, [None, 2], name="actions")
        self.quantile_midpoints = tf.placeholder(tf.float32, [None, self.num_atoms], name="quantile_midpoints")
        self.input = [self.actions, self.quantile_midpoints]

        # the output of the head is the N unordered quantile locations {theta_1, ..., theta_N}
        quantiles_locations = self.dense_layer(self.num_actions * self.num_atoms)(input_layer, name='output')
        quantiles_locations = tf.reshape(quantiles_locations, (tf.shape(quantiles_locations)[0], self.num_actions, self.num_atoms))
        self.output = quantiles_locations

        self.quantiles = tf.placeholder(tf.float32, shape=(None, self.num_atoms), name="quantiles")
        self.target = self.quantiles

        # only the quantiles of the taken action are taken into account
        quantiles_for_used_actions = tf.gather_nd(quantiles_locations, self.actions)

        # reorder the output quantiles and the target quantiles as a preparation step for calculating the loss
        # the output quantiles vector and the quantile midpoints are tiled as rows of a NxN matrix (N = num quantiles)
        # the target quantiles vector is tiled as column of a NxN matrix
        theta_i = tf.tile(tf.expand_dims(quantiles_for_used_actions, -1), [1, 1, self.num_atoms])
        T_theta_j = tf.tile(tf.expand_dims(self.target, -2), [1, self.num_atoms, 1])
        tau_i = tf.tile(tf.expand_dims(self.quantile_midpoints, -1), [1, 1, self.num_atoms])

        # Huber loss of T(theta_j) - theta_i
        error = T_theta_j - theta_i
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.huber_loss_interval)
        huber_loss = self.huber_loss_interval * (abs_error - quadratic) + 0.5 * quadratic ** 2

        # Quantile Huber loss
        quantile_huber_loss = tf.abs(tau_i - tf.cast(error < 0, dtype=tf.float32)) * huber_loss

        # Quantile regression loss (the probability for each quantile is 1/num_quantiles)
        quantile_regression_loss = tf.reduce_sum(quantile_huber_loss) / float(self.num_atoms)
        self.loss = quantile_regression_loss
        tf.losses.add_loss(self.loss)
