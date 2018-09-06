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

import copy
from typing import Dict

import numpy as np
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.architecture import TensorFlowArchitecture
from rl_coach.architectures.tensorflow_components.heads.head import HeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.middleware import MiddlewareParameters
from rl_coach.base_parameters import AgentParameters, EmbeddingMergerType
from rl_coach.core_types import PredictionType
from rl_coach.spaces import SpacesDefinition, PlanarMapsObservationSpace
from rl_coach.utils import get_all_subclasses, dynamic_import_and_instantiate_module_from_params


class GeneralTensorFlowNetwork(TensorFlowArchitecture):
    """
    A generalized version of all possible networks implemented using tensorflow.
    """
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str,
                 global_network=None, network_is_local: bool=True, network_is_trainable: bool=False):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """
        self.global_network = global_network
        self.network_is_local = network_is_local
        self.network_wrapper_name = name.split('/')[0]
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]
        self.num_heads_per_network = 1 if self.network_parameters.use_separate_networks_per_head else \
            len(self.network_parameters.heads_parameters)
        self.num_networks = 1 if not self.network_parameters.use_separate_networks_per_head else \
            len(self.network_parameters.heads_parameters)

        self.gradients_from_head_rescalers = []
        self.gradients_from_head_rescalers_placeholders = []
        self.update_head_rescaler_value_ops = []

        self.adaptive_learning_rate_scheme = None
        self.current_learning_rate = None

        # init network modules containers
        self.input_embedders = []
        self.output_heads = []
        super().__init__(agent_parameters, spaces, name, global_network,
                         network_is_local, network_is_trainable)

        def fill_return_types():
            ret_dict = {}
            for cls in get_all_subclasses(PredictionType):
                ret_dict[cls] = []
            components = self.input_embedders + [self.middleware] + self.output_heads
            for component in components:
                if not hasattr(component, 'return_type'):
                    raise ValueError("{} has no return_type attribute. This should not happen.")
                if component.return_type is not None:
                    ret_dict[component.return_type].append(component)

            return ret_dict

        self.available_return_types = fill_return_types()

    def predict_with_prediction_type(self, states: Dict[str, np.ndarray],
                                     prediction_type: PredictionType) -> Dict[str, np.ndarray]:
        """
        Search for a component[s] which has a return_type set to the to the requested PredictionType, and get
        predictions for it.

        :param states: The input states to the network.
        :param prediction_type: The requested PredictionType to look for in the network components
        :return: A dictionary with predictions for all components matching the requested prediction type
        """

        ret_dict = {}
        for component in self.available_return_types[prediction_type]:
            ret_dict[component] = self.predict(inputs=states, outputs=component.output)

        return ret_dict

    @staticmethod
    def get_activation_function(activation_function_string: str):
        """
        Map the activation function from a string to the tensorflow framework equivalent
        :param activation_function_string: the type of the activation function
        :return: the tensorflow activation function
        """
        activation_functions = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'elu': tf.nn.elu,
            'selu': tf.nn.selu,
            'leaky_relu': tf.nn.leaky_relu,
            'none': None
        }
        assert activation_function_string in activation_functions.keys(), \
            "Activation function must be one of the following {}. instead it was: {}"\
                .format(activation_functions.keys(), activation_function_string)
        return activation_functions[activation_function_string]

    def get_input_embedder(self, input_name: str, embedder_params: InputEmbedderParameters):
        """
        Given an input embedder parameters class, creates the input embedder and returns it
        :param input_name: the name of the input to the embedder (used for retrieving the shape). The input should
                           be a value within the state or the action.
        :param embedder_params: the parameters of the class of the embedder
        :return: the embedder instance
        """
        allowed_inputs = copy.copy(self.spaces.state.sub_spaces)
        allowed_inputs["action"] = copy.copy(self.spaces.action)
        allowed_inputs["goal"] = copy.copy(self.spaces.goal)

        if input_name not in allowed_inputs.keys():
            raise ValueError("The key for the input embedder ({}) must match one of the following keys: {}"
                             .format(input_name, allowed_inputs.keys()))

        type = "vector"
        if isinstance(allowed_inputs[input_name], PlanarMapsObservationSpace):
            type = "image"

        embedder_path = 'rl_coach.architectures.tensorflow_components.embedders.' + embedder_params.path[type]
        embedder_params_copy = copy.copy(embedder_params)
        embedder_params_copy.activation_function = self.get_activation_function(embedder_params.activation_function)
        embedder_params_copy.input_rescaling = embedder_params_copy.input_rescaling[type]
        embedder_params_copy.input_offset = embedder_params_copy.input_offset[type]
        embedder_params_copy.name = input_name
        module = dynamic_import_and_instantiate_module_from_params(embedder_params_copy,
                                                                   path=embedder_path,
                                                                   positional_args=[allowed_inputs[input_name].shape])
        return module

    def get_middleware(self, middleware_params: MiddlewareParameters):
        """
        Given a middleware type, creates the middleware and returns it
        :param middleware_params: the paramaeters of the middleware class
        :return: the middleware instance
        """
        middleware_params_copy = copy.copy(middleware_params)
        middleware_params_copy.activation_function = self.get_activation_function(middleware_params.activation_function)
        module = dynamic_import_and_instantiate_module_from_params(middleware_params_copy)
        return module

    def get_output_head(self, head_params: HeadParameters, head_idx: int, loss_weight: float=1.):
        """
        Given a head type, creates the head and returns it
        :param head_params: the parameters of the head to create
        :param head_type: the path to the class of the head under the embedders directory or a full path to a head class.
                          the path should be in the following structure: <module_path>:<class_path>
        :param head_idx: the head index
        :param loss_weight: the weight to assign for the embedders loss
        :return: the head
        """

        head_params_copy = copy.copy(head_params)
        head_params_copy.activation_function = self.get_activation_function(head_params_copy.activation_function)
        return dynamic_import_and_instantiate_module_from_params(head_params_copy, extra_kwargs={
            'agent_parameters': self.ap, 'spaces': self.spaces, 'network_name': self.network_wrapper_name,
            'head_idx': head_idx, 'loss_weight': loss_weight, 'is_local': self.network_is_local})

    def get_model(self):
        # validate the configuration
        if len(self.network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(self.network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if self.network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        if len(self.network_parameters.loss_weights) == 0:
            raise ValueError("At least one loss weight should be defined")

        if len(self.network_parameters.heads_parameters) != len(self.network_parameters.loss_weights):
            raise ValueError("Number of loss weights should match the number of output types")

        for network_idx in range(self.num_networks):
            with tf.variable_scope('network_{}'.format(network_idx)):

                ####################
                # Input Embeddings #
                ####################

                state_embedding = []
                for input_name in sorted(self.network_parameters.input_embedders_parameters):
                    input_type = self.network_parameters.input_embedders_parameters[input_name]
                    # get the class of the input embedder
                    input_embedder = self.get_input_embedder(input_name, input_type)
                    self.input_embedders.append(input_embedder)

                    # input placeholders are reused between networks. on the first network, store the placeholders
                    # generated by the input_embedders in self.inputs. on the rest of the networks, pass
                    # the existing input_placeholders into the input_embedders.
                    if network_idx == 0:
                        input_placeholder, embedding = input_embedder()
                        self.inputs[input_name] = input_placeholder
                    else:
                        input_placeholder, embedding = input_embedder(self.inputs[input_name])

                    state_embedding.append(embedding)

                ##########
                # Merger #
                ##########

                if len(state_embedding) == 1:
                    state_embedding = state_embedding[0]
                else:
                    if self.network_parameters.embedding_merger_type == EmbeddingMergerType.Concat:
                        state_embedding = tf.concat(state_embedding, axis=-1, name="merger")
                    elif self.network_parameters.embedding_merger_type == EmbeddingMergerType.Sum:
                        state_embedding = tf.add_n(state_embedding, name="merger")

                ##############
                # Middleware #
                ##############

                self.middleware = self.get_middleware(self.network_parameters.middleware_parameters)
                _, self.state_embedding = self.middleware(state_embedding)

                ################
                # Output Heads #
                ################

                head_count = 0
                for head_idx in range(self.num_heads_per_network):
                    for head_copy_idx in range(self.network_parameters.num_output_head_copies):
                        if self.network_parameters.use_separate_networks_per_head:
                            # if we use separate networks per head, then the head type corresponds top the network idx
                            head_type_idx = network_idx
                            head_count = network_idx
                        else:
                            # if we use a single network with multiple embedders, then the head type is the current head idx
                            head_type_idx = head_idx

                        # create output head and add it to the output heads list
                        self.output_heads.append(
                            self.get_output_head(self.network_parameters.heads_parameters[head_type_idx],
                                                 head_idx*self.network_parameters.num_output_head_copies + head_copy_idx,
                                                 self.network_parameters.loss_weights[head_type_idx])
                        )

                        # rescale the gradients from the head
                        self.gradients_from_head_rescalers.append(
                            tf.get_variable('gradients_from_head_{}-{}_rescalers'.format(head_idx, head_copy_idx),
                                            initializer=float(
                                                self.network_parameters.rescale_gradient_from_head_by_factor[head_count]
                                            ),
                                            dtype=tf.float32))

                        self.gradients_from_head_rescalers_placeholders.append(
                            tf.placeholder('float',
                                           name='gradients_from_head_{}-{}_rescalers'.format(head_type_idx, head_copy_idx)))

                        self.update_head_rescaler_value_ops.append(self.gradients_from_head_rescalers[head_count].assign(
                            self.gradients_from_head_rescalers_placeholders[head_count]))

                        head_input = (1-self.gradients_from_head_rescalers[head_count]) * tf.stop_gradient(self.state_embedding) + \
                                     self.gradients_from_head_rescalers[head_count] * self.state_embedding

                        # build the head
                        if self.network_is_local:
                            output, target_placeholder, input_placeholders, importance_weight_ph = \
                                self.output_heads[-1](head_input)

                            self.targets.extend(target_placeholder)
                            self.importance_weights.extend(importance_weight_ph)
                        else:
                            output, input_placeholders = self.output_heads[-1](head_input)

                        self.outputs.extend(output)
                        # TODO: use head names as well
                        for placeholder_index, input_placeholder in enumerate(input_placeholders):
                            self.inputs['output_{}_{}'.format(head_type_idx, placeholder_index)] = input_placeholder

                        head_count += 1

        # Losses
        self.losses = tf.losses.get_losses(self.full_name)
        self.losses += tf.losses.get_regularization_losses(self.full_name)
        self.total_loss = tf.losses.compute_weighted_loss(self.losses, scope=self.full_name)
        # tf.summary.scalar('total_loss', self.total_loss)

        # Learning rate
        if self.network_parameters.learning_rate_decay_rate != 0:
            self.adaptive_learning_rate_scheme = \
                tf.train.exponential_decay(
                    self.network_parameters.learning_rate,
                    self.global_step,
                    decay_steps=self.network_parameters.learning_rate_decay_steps,
                    decay_rate=self.network_parameters.learning_rate_decay_rate,
                    staircase=True)

            self.current_learning_rate = self.adaptive_learning_rate_scheme
        else:
            self.current_learning_rate = self.network_parameters.learning_rate

        # Optimizer
        if self.distributed_training and self.network_is_local and self.network_parameters.shared_optimizer:
            # distributed training + is a local network + optimizer shared -> take the global optimizer
            self.optimizer = self.global_network.optimizer
        elif (self.distributed_training and self.network_is_local and not self.network_parameters.shared_optimizer) \
                or self.network_parameters.shared_optimizer or not self.distributed_training:
            # distributed training + is a global network + optimizer shared
            # OR
            # distributed training + is a local network + optimizer not shared
            # OR
            # non-distributed training
            # -> create an optimizer

            if self.network_parameters.optimizer_type == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning_rate,
                                                        beta1=self.network_parameters.adam_optimizer_beta1,
                                                        beta2=self.network_parameters.adam_optimizer_beta2,
                                                        epsilon=self.network_parameters.optimizer_epsilon)
            elif self.network_parameters.optimizer_type == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.current_learning_rate,
                                                           decay=self.network_parameters.rms_prop_optimizer_decay,
                                                           epsilon=self.network_parameters.optimizer_epsilon)
            elif self.network_parameters.optimizer_type == 'LBFGS':
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B',
                                                                        options={'maxiter': 25})
            else:
                raise Exception("{} is not a valid optimizer type".format(self.network_parameters.optimizer_type))


