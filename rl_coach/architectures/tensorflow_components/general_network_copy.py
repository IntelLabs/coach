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
from typing import Dict, List, Union, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras

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


from rl_coach.architectures.tensorflow_components.general_model import GeneralModel
from rl_coach.architectures.tensorflow_components.general_loss import GeneralLoss



class SingleModel(keras.Model):
    """
    Block that connects a single embedder, with middleware and one to multiple heads
    """
    def __init__(self,
                 network_is_local: bool,
                 network_name: str,
                 agent_parameters: AgentParameters,
                 in_emb_param_dict: {str: InputEmbedderParameters},
                 embedding_merger_type: EmbeddingMergerType,
                 middleware_param: MiddlewareParameters,
                 head_param_list: [HeadParameters],
                 head_type_idx_start: int,
                 spaces: SpacesDefinition,
                 *args, **kwargs):
        """
        :param network_is_local: True if network is local
        :param network_name: name of the network
        :param agent_parameters: agent parameters
        :param in_emb_param_dict: dictionary of embedder name to embedding parameters
        :param embedding_merger_type: type of merging output of embedders: concatenate or sum
        :param middleware_param: middleware parameters
        :param head_param_list: list of head parameters, one per head type
        :param head_type_idx_start: start index for head type index counting
        :param spaces: state and action space definition
        """
        super(SingleModel, self).__init__(*args, **kwargs)

        self._embedding_merger_type = embedding_merger_type
        self._input_embedders = list()
        self._output_heads = list()

        for input_name in sorted(in_emb_param_dict):
            input_type = in_emb_param_dict[input_name]
            input_embedder = _get_input_embedder(spaces, input_name, input_type)
            self.register_child(input_embedder)
            self._input_embedders.append(input_embedder)

        self.middleware = _get_middleware(middleware_param)

        for i, head_param in enumerate(head_param_list):
            for head_copy_idx in range(head_param.num_output_head_copies):
                # create output head and add it to the output heads list
                output_head = ScaledGradHead(
                    head_index=(head_type_idx_start + i) * head_param.num_output_head_copies + head_copy_idx,
                    head_type_index=head_type_idx_start + i,
                    network_name=network_name,
                    spaces=spaces,
                    network_is_local=network_is_local,
                    agent_params=agent_parameters,
                    head_params=head_param)
                self.register_child(output_head)
                self._output_heads.append(output_head)

    def hybrid_forward(self, F, *inputs: Union[NDArray, Symbol]) -> Tuple[Union[NDArray, Symbol], ...]:
        """ Overrides gluon.HybridBlock.hybrid_forward
        :param nd or sym F: ndarray or symbol block
        :param inputs: model inputs, one for each embedder
        :return: head outputs in a tuple
        """
        # Input Embeddings
        state_embedding = list()
        for input, embedder in zip(inputs, self._input_embedders):
            state_embedding.append(embedder(input))

        # Merger
        if len(state_embedding) == 1:
            state_embedding = state_embedding[0]
        else:
            if self._embedding_merger_type == EmbeddingMergerType.Concat:
                state_embedding = F.concat(*state_embedding, dim=1, name='merger')  # NC or NCHW layout
            elif self._embedding_merger_type == EmbeddingMergerType.Sum:
                state_embedding = F.add_n(*state_embedding, name='merger')

        # Middleware
        state_embedding = self.middleware(state_embedding)

        # Head
        outputs = tuple()
        for head in self._output_heads:
            out = head(state_embedding)
            if not isinstance(out, tuple):
                out = (out,)
            outputs += out

        return outputs

    @property
    def input_embedders(self) -> List[HybridBlock]:
        """
        :return: list of input embedders
        """
        return self._input_embedders

    @property
    def output_heads(self) -> List[Head]:
        """
        :return: list of output heads
        """
        return [h.head for h in self._output_heads]




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
            with tf.compat.v1.variable_scope(variable_scope, auxiliary_name_scope=False) as vs:
                with tf.compat.v1.name_scope(vs.original_name_scope):
                    return construct_on_device()
        else:
            with tf.compat.v1.variable_scope(variable_scope, auxiliary_name_scope=True) as vs:
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


        #self.model = GeneralModel(agent_parameters, spaces, name)
        self.loss = GeneralLoss()


        super().__init__(agent_parameters, spaces, name, global_network,
                         network_is_local, network_is_trainable)

        self.available_return_types = self._available_return_types()
        self.is_training = None





    def _available_return_types(self):

        ret_dict = {cls: [] for cls in get_all_subclasses(PredictionType)}

        #components = self.input_embedders + [self.middleware] + self.output_heads
        components = self.dnn_model.input_embedders + [self.dnn_model.middleware] + self.dnn_model.output_heads

        for component in components:
            if not hasattr(component, 'return_type'):
                raise ValueError((
                    "{} has no return_type attribute. Without this, it is "
                    "unclear how this component should be used."
                ).format(component))

            if component.return_type is not None:
                ret_dict[component.return_type].append(component)

        return ret_dict

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


    def get_model(self) -> Callable:

        # DEBUG
        obs = np.array([1., 3., -44., 4.])
        obs_batch = tf.expand_dims(obs, 0)
        model = GeneralModel(self.ap, self.spaces, self.name)
        model.build(input_shape=(None, 4))
        #model(obs_batch)
        self.optimizer = self.get_optimizer()
        #return GeneralModel(self.ap, self.spaces, self.name)
        return model

    def get_optimizer(self) -> Callable:

        # Learning rate
        if self.network_parameters.learning_rate_decay_rate != 0:
            self.adaptive_learning_rate_scheme = \
                tf.compat.v1.train.exponential_decay(
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
            optimizer = self.global_network.optimizer

        elif (self.distributed_training and self.network_is_local and not self.network_parameters.shared_optimizer) \
                or self.network_parameters.shared_optimizer or not self.distributed_training:
            # distributed training + is a global network + optimizer shared
            # OR
            # distributed training + is a local network + optimizer not shared
            # OR
            # non-distributed training
            # -> create an optimizer
            if self.network_parameters.optimizer_type == 'Adam':

                optimizer = keras.optimizers.Adam(
                    lr=self.current_learning_rate,
                    beta_1=self.network_parameters.adam_optimizer_beta1,
                    beta_2=self.network_parameters.adam_optimizer_beta2,
                    epsilon=self.network_parameters.optimizer_epsilon)

            elif self.network_parameters.optimizer_type == 'RMSProp':
                optimizer = keras.optimizers.RMSprop(
                    lr=self.current_learning_rate,
                    decay=self.network_parameters.rms_prop_optimizer_decay,
                    epsilon=self.network_parameters.optimizer_epsilon)

            elif self.network_parameters.optimizer_type == 'LBFGS':
                raise NotImplementedError(' Could not find updated LBFGS implementation')  # TODO: Dan to update function
                # Dan manual fix
                # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B',
                #                                                                          options={'maxiter': 25})
            else:
                raise Exception("{} is not a valid optimizer type".format(self.network_parameters.optimizer_type))

        return optimizer

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
