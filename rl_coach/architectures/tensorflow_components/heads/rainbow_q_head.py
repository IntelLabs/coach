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
import numpy as np
from rl_coach.architectures.tensorflow_components.heads import QHead
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class RainbowQHead(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.num_actions = len(self.spaces.action.actions)
        self.num_atoms = agent_parameters.algorithm.atoms
        self.name = 'rainbow_q_values_head'
        self.z_values = tf.cast(tf.constant(np.linspace(self.ap.algorithm.v_min, self.ap.algorithm.v_max,
                                                        self.ap.algorithm.atoms), dtype=tf.float32), dtype=tf.float64)
        self.loss_type = []

    def _build_module(self, input_layer):
        # state value tower - V
        with tf.variable_scope("state_value"):
            state_value = self.dense_layer(512)(input_layer, activation=self.activation_function, name='fc1')
            state_value = self.dense_layer(self.num_atoms)(state_value, name='fc2')
            state_value = tf.expand_dims(state_value, axis=1)

        # action advantage tower - A
        with tf.variable_scope("action_advantage"):
            action_advantage = self.dense_layer(512)(input_layer, activation=self.activation_function, name='fc1')
            action_advantage = self.dense_layer(self.num_actions * self.num_atoms)(action_advantage, name='fc2')
            action_advantage = tf.reshape(action_advantage, (tf.shape(input_layer)[0], self.num_actions,
                                                             self.num_atoms))
            action_mean = tf.reduce_mean(action_advantage, axis=1, keepdims=True)
            action_advantage = action_advantage - action_mean

        # merge to state-action value function Q
        values_distribution = tf.add(state_value, action_advantage, name='output')

        # softmax on atoms dimension
        self.output = tf.nn.softmax(values_distribution)

        # calculate cross entropy loss
        self.distributions = tf.placeholder(tf.float32, shape=(None, self.num_actions, self.num_atoms),
                                            name="distributions")
        self.target = self.distributions
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=values_distribution)
        tf.losses.add_loss(self.loss)

        self.q_values = tf.tensordot(tf.cast(self.output, tf.float64), self.z_values, 1)

        # used in batch-rl to estimate a probablity distribution over actions
        self.softmax = self.add_softmax_with_temperature()

    def __str__(self):
        result = [
            "State Value Stream - V",
            "\tDense (num outputs = 512)",
            "\tDense (num outputs = {})".format(self.num_atoms),
            "Action Advantage Stream - A",
            "\tDense (num outputs = 512)",
            "\tDense (num outputs = {})".format(self.num_actions * self.num_atoms),
            "\tReshape (new size = {} x {})".format(self.num_actions, self.num_atoms),
            "\tSubtract(A, Mean(A))".format(self.num_actions),
            "Add (V, A)",
            "Softmax"
        ]
        return '\n'.join(result)
