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

from types import ModuleType
from rl_coach.architectures.tensorflow_components.embedders import ImageEmbedder, TensorEmbedder, VectorEmbedder
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters, LSTMMiddlewareParameters
from rl_coach.architectures.tensorflow_components.middlewares import FCMiddleware, LSTMMiddleware
from rl_coach.architectures.tensorflow_components.heads import Head, QHead
from rl_coach.architectures.head_parameters import QHeadParameters
from itertools import chain
from rl_coach.base_parameters import NetworkParameters
from rl_coach.architectures.tensorflow_components.architecture import TensorFlowArchitecture


class GeneralLoss(keras.losses.Loss):
    def __init__(self, loss_type='MeanSquaredError', **kwargs):
        self.loss_type = loss_type
        self.loss_fn = keras.losses.get(self.loss_type)
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "loss_type": self.loss_type}


class GeneralTensorFlowNetwork(TensorFlowArchitecture):
    """
    A generalized version of all possible networks implemented using tensorflow along with the optimizer and loss.
    """
    def construct(variable_scope: str, devices: List[str], *args, **kwargs) -> 'GeneralTensorFlowNetwork':
        """
        Construct a network class using the provided variable scope and on requested devices
        :param variable_scope: string specifying variable scope under which to create network variables
        :param devices: list of devices (can be list of Device objects, or string for TF distributed)
        :param args: all other arguments for class initializer
        :param kwargs: all other keyword arguments for class initializer
        :return: a GeneralTensorFlowNetwork object
        """
        # TODO: Dan place holder for distributed training in TensorFlow

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            generalized_network = GeneralTensorFlowNetwork(*args, **kwargs)
            loss = generalized_network.losses
            optimizer = generalized_network.optimizer
            generalized_network.model.compile(loss=loss, optimizer=optimizer)

        return generalized_network


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

    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 name: str,
                 global_network=None,
                 network_is_local: bool=True,
                 network_is_trainable: bool=False):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param devices: list of devices to run the network on
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """

        super().__init__(agent_parameters, spaces, name, global_network, network_is_local, network_is_trainable)

        self.global_network = global_network

        self.network_wrapper_name = name.split('/')[0]

        network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]

        if len(network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        if network_parameters.use_separate_networks_per_head:
            num_heads_per_network = 1
            num_networks = len(network_parameters.heads_parameters)
        else:
            num_heads_per_network = len(network_parameters.heads_parameters)
            num_networks = 1

        self.model = GeneralModel(
            num_networks=num_networks,
            num_heads_per_network=num_heads_per_network,
            network_is_local=network_is_local,
            network_name=self.network_wrapper_name,
            agent_parameters=agent_parameters,
            network_parameters=network_parameters,
            spaces=spaces)

        self.losses = GeneralLoss()
        self.optimizer = self.get_optimizer(network_parameters)
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]

    def get_optimizer(self, network_parameters):

        # callback = tf.keras.callbacks.LearningRateScheduler(
        #     (lambda lr, decay_rate, decay_steps, global_step: lr * (decay_rate ** (global_step / decay_steps))))
        # TODO: fix conditions in the if statement and add callback for learning rate scheduling
        if 0: #network_parameters.shared_optimizer:
            # Take the global optimizer
            optimizer = self.global_network.optimizer

        else:
            if network_parameters.optimizer_type == 'Adam':

                optimizer = keras.optimizers.Adam(
                    lr=network_parameters.learning_rate,
                    beta_1=network_parameters.adam_optimizer_beta1,
                    beta_2=network_parameters.adam_optimizer_beta2,
                    epsilon=network_parameters.optimizer_epsilon)

            elif network_parameters.optimizer_type == 'RMSProp':
                optimizer = keras.optimizers.RMSprop(
                    lr=network_parameters.learning_rate,
                    decay=network_parameters.rms_prop_optimizer_decay,
                    epsilon=network_parameters.optimizer_epsilon)

            elif network_parameters.optimizer_type == 'LBFGS':
                raise NotImplementedError(' Could not find updated LBFGS implementation')  # TODO: Dan to update function
            else:
                raise Exception("{} is not a valid optimizer type".format(self.network_parameters.optimizer_type))

        return optimizer





def _get_input_embedder(spaces: SpacesDefinition,
                        input_name: str,
                        embedder_params: InputEmbedderParameters) -> ModuleType:
    """
    Given an input embedder parameters class, creates the input embedder and returns it
    :param input_name: the name of the input to the embedder (used for retrieving the shape). The input should
                       be a value within the state or the action.
    :param embedder_params: the parameters of the class of the embedder
    :return: the embedder instance
    """
    allowed_inputs = copy.copy(spaces.state.sub_spaces)
    allowed_inputs["action"] = copy.copy(spaces.action)
    allowed_inputs["goal"] = copy.copy(spaces.goal)

    if input_name not in allowed_inputs.keys():
        raise ValueError("The key for the input embedder ({}) must match one of the following keys: {}"
                         .format(input_name, allowed_inputs.keys()))


    type = "vector"
    if isinstance(allowed_inputs[input_name], TensorObservationSpace):
        type = "tensor"
    elif isinstance(allowed_inputs[input_name], PlanarMapsObservationSpace):
        type = "image"

    embedder_params = copy.copy(embedder_params)
    embedder_params.name = input_name

    if type == 'vector':

        module = VectorEmbedder(input_size=allowed_inputs[input_name].shape,
                                activation_function=embedder_params.activation_function,
                                scheme=embedder_params.scheme,
                                batchnorm=embedder_params.batchnorm,
                                dropout_rate=embedder_params.dropout_rate,
                                name=embedder_params.name,
                                input_rescaling=embedder_params.input_rescaling[type],
                                input_offset=embedder_params.input_offset[type],
                                input_clipping=embedder_params.input_clipping,
                                is_training=embedder_params.is_training)
    elif type == 'image':
        module = ImageEmbedder(embedder_params)
    elif type == 'tensor':
        module = TensorEmbedder(embedder_params)
    else:
        raise KeyError('Unsupported embedder type: {}'.format(type))
    return module


def _get_middleware(middleware_params: MiddlewareParameters) -> ModuleType:
    """
    Given a middleware type, creates the middleware and returns it
    :param middleware_params: the paramaeters of the middleware class
    :return: the middleware instance
    """
    if isinstance(middleware_params, FCMiddlewareParameters):
        module = FCMiddleware(activation_function=middleware_params.activation_function,
                              scheme=middleware_params.scheme,
                              batchnorm=middleware_params.batchnorm,
                              dropout_rate=middleware_params.dropout_rate,
                              name=middleware_params.name,
                              is_training=middleware_params.is_training,
                              num_streams=middleware_params.num_streams)
    elif isinstance(middleware_params, LSTMMiddlewareParameters):
        module = LSTMMiddleware(middleware_params)
    else:
        raise KeyError('Unsupported middleware type: {}'.format(type(middleware_params)))

    return module


def _get_output_head(
        head_params: HeadParameters,
        head_idx: int,
        head_type_index: int,
        agent_params: AgentParameters,
        spaces: SpacesDefinition,
        network_name: str,
        is_local: bool) -> Head:
    """
    Given a head type, creates the head and returns it
    :param head_params: the parameters of the head to create
    :param head_idx: the head index
    :param head_type_index: the head type index (same index if head_param.num_output_head_copies>0)
    :param agent_params: agent parameters
    :param spaces: state and action space definitions
    :param network_name: name of the network
    :param is_local:
    :return: head block
    """

    if isinstance(head_params, QHeadParameters):
        module = QHead(
            agent_parameters=agent_params,
            spaces=spaces,
            network_name=network_name,
            head_type_idx=head_type_index,
            loss_weight=head_params.loss_weight,
            is_local=is_local,
            activation_function=head_params.activation_function,
            dense_layer=head_params.dense_layer)
    else:
        raise KeyError('Unsupported head type: {}'.format(type(head_params)))

    return module



class SingleWorkerModel(keras.Model):
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
        super(SingleWorkerModel, self).__init__(*args, **kwargs)

        self._embedding_merger_type = embedding_merger_type
        self._input_embedders = []
        self._output_heads = []

        for input_name in sorted(in_emb_param_dict):
            input_type = in_emb_param_dict[input_name]
            input_embedder = _get_input_embedder(spaces, input_name, input_type)
            self._input_embedders.append(input_embedder)

        self.middleware = _get_middleware(middleware_param)

        for i, head_param in enumerate(head_param_list):
            for head_copy_idx in range(head_param.num_output_head_copies):
                # create output head and add it to the output heads list
                head_idx = (head_type_idx_start + i) * head_param.num_output_head_copies + head_copy_idx
                output_head = _get_output_head(
                    head_idx=head_idx,
                    head_type_index=head_type_idx_start + i,
                    network_name=network_name,
                    spaces=spaces,
                    is_local=network_is_local,
                    agent_params=agent_parameters,
                    head_params=head_param)

                self._output_heads.append(output_head)
                #self._output_heads = output_head

    def call(self, inputs, **kwargs):
        """ Overrides tf.keras.call
        :param inputs: model inputs, one for each embedder
        :return: head outputs in a tuple
        """
        # Input Embeddings
        state_embedding = list()
        for input, embedder in zip(inputs, self._input_embedders):
            state_embedding.append(embedder(input))

        # for embedder in self.input_embedders:
        #     state_embedding.append(embedder(inputs))

        # Merger
        if len(state_embedding) == 1:
            # TODO: change to squeeze
            state_embedding = state_embedding[0]
        else:
            if self._embedding_merger_type == EmbeddingMergerType.Concat:
                #state_embedding = F.concat(*state_embedding, dim=1, name='merger')
                state_embedding = tf.keras.layers.Concatenate()(state_embedding)
            elif self._embedding_merger_type == EmbeddingMergerType.Sum:
                state_embedding = tf.keras.layers.Add()(state_embedding)
                #state_embedding = F.add_n(*state_embedding, name='merger')


        # Middleware
        state_embedding = self.middleware(state_embedding)

        # Head
        outputs = tuple()
        for head in self._output_heads:
            out = head(state_embedding)
            if not isinstance(out, tuple):
                out = (out,)
            outputs += out

        # Dan for debug
        #outputs = self._output_heads(state_embedding)
        return outputs

    @property
    def input_embedders(self):
        """
        :return: list of input embedders
        """
        return self._input_embedders

    @property
    def output_heads(self):
        """
        :return: list of output heads
        """
        return [h.head for h in self._output_heads]


class GeneralModel(keras.Model):
    """
    Block that creates multiple single models
    """
    def __init__(self,
                 num_networks: int,
                 num_heads_per_network: int,
                 network_is_local: bool,
                 network_name: str,
                 agent_parameters: AgentParameters,
                 network_parameters: NetworkParameters,
                 spaces: SpacesDefinition,
                 *args, **kwargs):
        """
        :param num_networks: number of networks to create
        :param num_heads_per_network: number of heads per network to create
        :param network_is_local: True if network is local
        :param network_name: name of the network
        :param agent_parameters: agent parameters
        :param network_parameters: network parameters
        :param spaces: state and action space definitions
        """
        super(GeneralModel, self).__init__(*args, **kwargs)

        self.nets = list()
        for network_idx in range(num_networks):
            head_type_idx_start = network_idx * num_heads_per_network
            head_type_idx_end = head_type_idx_start + num_heads_per_network
            net = SingleWorkerModel(
                head_type_idx_start=head_type_idx_start,
                network_name=network_name,
                network_is_local=network_is_local,
                agent_parameters=agent_parameters,
                in_emb_param_dict=network_parameters.input_embedders_parameters,
                embedding_merger_type=network_parameters.embedding_merger_type,
                middleware_param=network_parameters.middleware_parameters,
                head_param_list=network_parameters.heads_parameters[head_type_idx_start:head_type_idx_end],
                spaces=spaces)
            self.nets.append(net)
            self.single_model = self.nets[0]


    def call(self, inputs, **kwargs):
        """ Overrides tf.keras.call
        :param inputs: model inputs, one for each embedder. Passed to all networks.
        :return: head outputs in a tuple
        """
        # outputs = self.single_model(inputs)
        # outputs = tuple(outputs)
        outputs = []
        for net in self.nets:
            out = net(inputs)
            outputs.append(out)
            #outputs = out


        return outputs

    @property
    def output_heads(self) -> List[Head]:
        """ Return all heads in a single list
        Note: There is a one-to-one mapping between output_heads and losses
        :return: list of heads
        """
        return list(chain.from_iterable(net.output_heads for net in self.nets))

    # def losses(self) -> List[HeadLoss]:
    #     """ Construct loss blocks for network training
    #     Note: There is a one-to-one mapping between output_heads and losses
    #     :return: list of loss blocks
    #     """
    #     return [h.loss() for net in self.nets for h in net.output_heads]



