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
from rl_coach.architectures.tensorflow_components.heads.head import HeadParameters
from rl_coach.architectures.tensorflow_components.heads.q_head import QHead
from rl_coach.base_parameters import AgentParameters
from rl_coach.memories.non_episodic import differentiable_neural_dictionary
from rl_coach.spaces import SpacesDefinition


class DNDQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='dnd_q_head_params', dense_layer=Dense):
        super().__init__(parameterized_class=DNDQHead, activation_function=activation_function, name=name,
                         dense_layer=dense_layer)


class DNDQHead(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'dnd_q_values_head'
        self.DND_size = agent_parameters.algorithm.dnd_size
        self.DND_key_error_threshold = agent_parameters.algorithm.DND_key_error_threshold
        self.l2_norm_added_delta = agent_parameters.algorithm.l2_norm_added_delta
        self.new_value_shift_coefficient = agent_parameters.algorithm.new_value_shift_coefficient
        self.number_of_nn = agent_parameters.algorithm.number_of_knn
        self.ap = agent_parameters
        self.dnd_embeddings = [None] * self.num_actions
        self.dnd_values = [None] * self.num_actions
        self.dnd_indices = [None] * self.num_actions
        self.dnd_distances = [None] * self.num_actions
        if self.ap.memory.shared_memory:
            self.shared_memory_scratchpad = self.ap.task_parameters.shared_memory_scratchpad

    def _build_module(self, input_layer):
        if hasattr(self.ap.task_parameters, 'checkpoint_restore_dir') and self.ap.task_parameters.checkpoint_restore_dir:
            self.DND = differentiable_neural_dictionary.load_dnd(self.ap.task_parameters.checkpoint_restore_dir)
        else:
            self.DND = differentiable_neural_dictionary.QDND(
                self.DND_size, input_layer.get_shape()[-1], self.num_actions, self.new_value_shift_coefficient,
                key_error_threshold=self.DND_key_error_threshold,
                learning_rate=self.network_parameters.learning_rate,
                num_neighbors=self.number_of_nn,
                override_existing_keys=True)

        # Retrieve info from DND dictionary
        # We assume that all actions have enough entries in the DND
        self.output = tf.transpose([
            self._q_value(input_layer, action)
            for action in range(self.num_actions)
        ])

    def _q_value(self, input_layer, action):
        result = tf.py_func(self.DND.query,
                            [input_layer, action, self.number_of_nn],
                            [tf.float64, tf.float64, tf.int64])
        self.dnd_embeddings[action] = tf.to_float(result[0])
        self.dnd_values[action] = tf.to_float(result[1])
        self.dnd_indices[action] = result[2]

        # DND calculation
        square_diff = tf.square(self.dnd_embeddings[action] - tf.expand_dims(input_layer, 1))
        distances = tf.reduce_sum(square_diff, axis=2) + [self.l2_norm_added_delta]
        self.dnd_distances[action] = distances
        weights = 1.0 / distances
        normalised_weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
        q_value = tf.reduce_sum(self.dnd_values[action] * normalised_weights, axis=1)
        q_value.set_shape((None,))
        return q_value

    def _post_build(self):
        # DND gradients
        self.dnd_embeddings_grad = tf.gradients(self.loss[0], self.dnd_embeddings)
        self.dnd_values_grad = tf.gradients(self.loss[0], self.dnd_values)
