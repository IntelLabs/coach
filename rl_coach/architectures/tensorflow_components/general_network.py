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
from types import MethodType
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.middleware_parameters import MiddlewareParameters
from rl_coach.architectures.tensorflow_components.architecture import TensorFlowArchitecture
from rl_coach.architectures.tensorflow_components import utils
from rl_coach.base_parameters import AgentParameters, Device, DeviceType, EmbeddingMergerType
from rl_coach.core_types import PredictionType
from rl_coach.logger import screen
from rl_coach.spaces import SpacesDefinition, PlanarMapsObservationSpace, TensorObservationSpace
from rl_coach.utils import get_all_subclasses, dynamic_import_and_instantiate_module_from_params, indent_string


class GeneralTensorFlowNetwork(TensorFlowArchitecture):
    """
    A generalized version of all possible networks implemented using tensorflow.
    """
    # dictionary of variable-scope name to variable-scope object to prevent tensorflow from
    # creating a new auxiliary variable scope even when name is properly specified
    variable_scopes_dict = dict()

    @staticmethod
    def construct(variable_scope: str, devices: List[str], *args, **kwargs) -> 'GeneralTensorFlowNetwork':
        """
        Construct a network class using the provided variable scope and on requested devices
        :param variable_scope: string specifying variable scope under which to create network variables
        :param devices: list of devices (can be list of Device objects, or string for TF distributed)
        :param args: all other arguments for class initializer
        :param kwargs: all other keyword arguments for class initializer
        :return: a GeneralTensorFlowNetwork object
        """
        if len(devices) > 1:
            screen.warning("Tensorflow implementation only support a single device. Using {}".format(devices[0]))

        def construct_on_device():
            with tf.device(GeneralTensorFlowNetwork._tf_device(devices[0])):
                return GeneralTensorFlowNetwork(*args, **kwargs)

        # If variable_scope is in our dictionary, then this is not the first time that this variable_scope
        # is being used with construct(). So to avoid TF adding an incrementing number to the end of the
        # variable_scope to uniquify it, we have to both pass the previous variable_scope object to the new
        # variable_scope() call and also recover the name space using name_scope
        if variable_scope in GeneralTensorFlowNetwork.variable_scopes_dict:
            variable_scope = GeneralTensorFlowNetwork.variable_scopes_dict[variable_scope]
            with tf.variable_scope(variable_scope, auxiliary_name_scope=False) as vs:
                with tf.name_scope(vs.original_name_scope):
                    return construct_on_device()
        else:
            with tf.variable_scope(variable_scope, auxiliary_name_scope=True) as vs:
                # Add variable_scope object to dictionary for next call to construct
                GeneralTensorFlowNetwork.variable_scopes_dict[variable_scope] = vs
                return construct_on_device()

    @staticmethod
    def _tf_device(device: Union[str, MethodType, Device]) -> str:
        """
        Convert device to tensorflow-specific device representation
        :param device: either a specific string or method (used in distributed mode) which is returned without
            any change or a Device type, which will be converted to a string
        :return: tensorflow-specific string for device
        """
        if isinstance(device, str) or isinstance(device, MethodType):
            return device
        elif isinstance(device, Device):
            if device.device_type == DeviceType.CPU:
                return "/cpu:0"
            elif device.device_type == DeviceType.GPU:
                return "/device:GPU:{}".format(device.index)
            else:
                raise ValueError("Invalid device_type: {}".format(device.device_type))
        else:
            raise ValueError("Invalid device instance type: {}".format(type(device)))

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
        self.is_training = None

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

        mod_names = {'image': 'ImageEmbedder', 'vector': 'VectorEmbedder', 'tensor': 'TensorEmbedder'}

        emb_type = "vector"
        if isinstance(allowed_inputs[input_name], TensorObservationSpace):
            emb_type = "tensor"
        elif isinstance(allowed_inputs[input_name], PlanarMapsObservationSpace):
            emb_type = "image"

        embedder_path = 'rl_coach.architectures.tensorflow_components.embedders:' + mod_names[emb_type]
        embedder_params_copy = copy.copy(embedder_params)
        embedder_params_copy.activation_function = utils.get_activation_function(embedder_params.activation_function)
        embedder_params_copy.input_rescaling = embedder_params_copy.input_rescaling[emb_type]
        embedder_params_copy.input_offset = embedder_params_copy.input_offset[emb_type]
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
        mod_name = middleware_params.parameterized_class_name
        middleware_path = 'rl_coach.architectures.tensorflow_components.middlewares:' + mod_name
        middleware_params_copy = copy.copy(middleware_params)
        middleware_params_copy.activation_function = utils.get_activation_function(middleware_params.activation_function)
        module = dynamic_import_and_instantiate_module_from_params(middleware_params_copy, path=middleware_path)
        return module

    def get_output_head(self, head_params: HeadParameters, head_idx: int):
        """
        Given a head type, creates the head and returns it
        :param head_params: the parameters of the head to create
        :param head_idx: the head index
        :return: the head
        """
        mod_name = head_params.parameterized_class_name
        head_path = 'rl_coach.architectures.tensorflow_components.heads:' + mod_name
        head_params_copy = copy.copy(head_params)
        head_params_copy.activation_function = utils.get_activation_function(head_params_copy.activation_function)
        return dynamic_import_and_instantiate_module_from_params(head_params_copy, path=head_path, extra_kwargs={
            'agent_parameters': self.ap, 'spaces': self.spaces, 'network_name': self.network_wrapper_name,
            'head_idx': head_idx, 'is_local': self.network_is_local})

    def get_model(self):
        # validate the configuration
        if len(self.network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(self.network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if self.network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        # ops for defining the training / testing phase
        self.is_training = tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.is_training_placeholder = tf.placeholder("bool")
        self.assign_is_training = tf.assign(self.is_training, self.is_training_placeholder)

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

                    if self.network_parameters.use_separate_networks_per_head:
                        # if we use separate networks per head, then the head type corresponds to the network idx
                        head_type_idx = network_idx
                        head_count = network_idx
                    else:
                        # if we use a single network with multiple embedders, then the head type is the current head idx
                        head_type_idx = head_idx
                    head_params = self.network_parameters.heads_parameters[head_type_idx]

                    for head_copy_idx in range(head_params.num_output_head_copies):
                        # create output head and add it to the output heads list
                        self.output_heads.append(
                            self.get_output_head(head_params,
                                                 head_idx*head_params.num_output_head_copies + head_copy_idx)
                        )

                        # rescale the gradients from the head
                        self.gradients_from_head_rescalers.append(
                            tf.get_variable('gradients_from_head_{}-{}_rescalers'.format(head_idx, head_copy_idx),
                                            initializer=float(head_params.rescale_gradient_from_head_by_factor),
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
        self.total_loss = tf.reduce_sum(self.losses)
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

    def __str__(self):
        result = []

        for network in range(self.num_networks):
            network_structure = []

            # embedder
            for embedder in self.input_embedders:
                network_structure.append("Input Embedder: {}".format(embedder.name))
                network_structure.append(indent_string(str(embedder)))

            if len(self.input_embedders) > 1:
                network_structure.append("{} ({})".format(self.network_parameters.embedding_merger_type.name,
                                               ", ".join(["{} embedding".format(e.name) for e in self.input_embedders])))

            # middleware
            network_structure.append("Middleware:")
            network_structure.append(indent_string(str(self.middleware)))

            # head
            if self.network_parameters.use_separate_networks_per_head:
                heads = range(network, network+1)
            else:
                heads = range(0, len(self.output_heads))

            for head_idx in heads:
                head = self.output_heads[head_idx]
                head_params = self.network_parameters.heads_parameters[head_idx]
                if head_params.num_output_head_copies > 1:
                    network_structure.append("Output Head: {} (num copies = {})".format(head.name, head_params.num_output_head_copies))
                else:
                    network_structure.append("Output Head: {}".format(head.name))
                    network_structure.append(indent_string(str(head)))

            # finalize network
            if self.num_networks > 1:
                result.append("Sub-network for head: {}".format(self.output_heads[network].name))
                result.append(indent_string('\n'.join(network_structure)))
            else:
                result.append('\n'.join(network_structure))

        result = '\n'.join(result)
        return result
