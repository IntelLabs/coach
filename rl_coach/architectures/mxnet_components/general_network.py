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
from itertools import chain
from typing import List, Tuple, Union
from types import ModuleType

import numpy as np
import mxnet as mx
from mxnet import nd, sym
from mxnet.gluon import HybridBlock
from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol

from rl_coach.base_parameters import NetworkParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import HeadParameters, PPOHeadParameters
from rl_coach.architectures.head_parameters import PPOVHeadParameters, VHeadParameters, QHeadParameters
from rl_coach.architectures.middleware_parameters import MiddlewareParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters, LSTMMiddlewareParameters
from rl_coach.architectures.mxnet_components.architecture import MxnetArchitecture
from rl_coach.architectures.mxnet_components.embedders import ImageEmbedder, TensorEmbedder, VectorEmbedder
from rl_coach.architectures.mxnet_components.heads import Head, HeadLoss, PPOHead, PPOVHead, VHead, QHead
from rl_coach.architectures.mxnet_components.middlewares import FCMiddleware, LSTMMiddleware
from rl_coach.architectures.mxnet_components import utils
from rl_coach.base_parameters import AgentParameters, Device, DeviceType, EmbeddingMergerType
from rl_coach.spaces import SpacesDefinition, PlanarMapsObservationSpace, TensorObservationSpace


class GeneralMxnetNetwork(MxnetArchitecture):
    """
    A generalized version of all possible networks implemented using mxnet.
    """
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
        return GeneralMxnetNetwork(*args, devices=[GeneralMxnetNetwork._mx_device(d) for d in devices], **kwargs)

    @staticmethod
    def _mx_device(device: Union[str, Device]) -> mx.Context:
        """
        Convert device to tensorflow-specific device representation
        :param device: either a specific string (used in distributed mode) which is returned without
            any change or a Device type
        :return: tensorflow-specific string for device
        """
        if isinstance(device, Device):
            if device.device_type == DeviceType.CPU:
                return mx.cpu()
            elif device.device_type == DeviceType.GPU:
                return mx.gpu(device.index)
            else:
                raise ValueError("Invalid device_type: {}".format(device.device_type))
        else:
            raise ValueError("Invalid device instance type: {}".format(type(device)))

    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 devices: List[mx.Context],
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
        self.network_wrapper_name = name.split('/')[0]
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]
        if self.network_parameters.use_separate_networks_per_head:
            self.num_heads_per_network = 1
            self.num_networks = len(self.network_parameters.heads_parameters)
        else:
            self.num_heads_per_network = len(self.network_parameters.heads_parameters)
            self.num_networks = 1

        super().__init__(agent_parameters, spaces, devices, name, global_network,
                         network_is_local, network_is_trainable)

    def construct_model(self):
        # validate the configuration
        if len(self.network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(self.network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if self.network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        self.model = GeneralModel(
            num_networks=self.num_networks,
            num_heads_per_network=self.num_heads_per_network,
            network_is_local=self.network_is_local,
            network_name=self.network_wrapper_name,
            agent_parameters=self.ap,
            network_parameters=self.network_parameters,
            spaces=self.spaces)

        self.losses = self.model.losses()

        # Learning rate
        lr_scheduler = None
        if self.network_parameters.learning_rate_decay_rate != 0:
            lr_scheduler = mx.lr_scheduler.FactorScheduler(
                step=self.network_parameters.learning_rate_decay_steps,
                factor=self.network_parameters.learning_rate_decay_rate)

        # Optimizer
        # FIXME Does this code for distributed training make sense?
        if self.distributed_training and self.network_is_local and self.network_parameters.shared_optimizer:
            # distributed training + is a local network + optimizer shared -> take the global optimizer
            self.optimizer = self.global_network.optimizer
        elif (self.distributed_training and self.network_is_local and not self.network_parameters.shared_optimizer)\
                or self.network_parameters.shared_optimizer or not self.distributed_training:

            if self.network_parameters.optimizer_type == 'Adam':
                self.optimizer = mx.optimizer.Adam(
                    learning_rate=self.network_parameters.learning_rate,
                    beta1=self.network_parameters.adam_optimizer_beta1,
                    beta2=self.network_parameters.adam_optimizer_beta2,
                    epsilon=self.network_parameters.optimizer_epsilon,
                    lr_scheduler=lr_scheduler)
            elif self.network_parameters.optimizer_type == 'RMSProp':
                self.optimizer = mx.optimizer.RMSProp(
                    learning_rate=self.network_parameters.learning_rate,
                    gamma1=self.network_parameters.rms_prop_optimizer_decay,
                    epsilon=self.network_parameters.optimizer_epsilon,
                    lr_scheduler=lr_scheduler)
            elif self.network_parameters.optimizer_type == 'LBFGS':
                raise NotImplementedError('LBFGS optimizer not implemented')
            else:
                raise Exception("{} is not a valid optimizer type".format(self.network_parameters.optimizer_type))

    @property
    def output_heads(self):
        return self.model.output_heads


def _get_activation(activation_function_string: str):
    """
    Map the activation function from a string to the mxnet framework equivalent
    :param activation_function_string: the type of the activation function
    :return: mxnet activation function string
    """
    return utils.get_mxnet_activation_name(activation_function_string)


def _sanitize_activation(params: Union[InputEmbedderParameters, MiddlewareParameters, HeadParameters]) ->\
        Union[InputEmbedderParameters, MiddlewareParameters, HeadParameters]:
    """
    Change activation function to the mxnet specific value
    :param params: any parameter that has activation_function property
    :return: copy of params with activation function correctly set
    """
    params_copy = copy.copy(params)
    params_copy.activation_function = _get_activation(params.activation_function)
    return params_copy


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

    def sanitize_params(params: InputEmbedderParameters):
        params_copy = _sanitize_activation(params)
        # params_copy.input_rescaling = params_copy.input_rescaling[type]
        # params_copy.input_offset = params_copy.input_offset[type]
        params_copy.name = input_name
        return params_copy

    embedder_params = sanitize_params(embedder_params)
    if type == 'vector':
        module = VectorEmbedder(embedder_params)
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
    middleware_params = _sanitize_activation(middleware_params)
    if isinstance(middleware_params, FCMiddlewareParameters):
        module = FCMiddleware(middleware_params)
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
    head_params = _sanitize_activation(head_params)
    if isinstance(head_params, PPOHeadParameters):
        module = PPOHead(
            agent_parameters=agent_params,
            spaces=spaces,
            network_name=network_name,
            head_type_idx=head_type_index,
            loss_weight=head_params.loss_weight,
            is_local=is_local,
            activation_function=head_params.activation_function,
            dense_layer=head_params.dense_layer)
    elif isinstance(head_params, VHeadParameters):
        module = VHead(
            agent_parameters=agent_params,
            spaces=spaces,
            network_name=network_name,
            head_type_idx=head_type_index,
            loss_weight=head_params.loss_weight,
            is_local=is_local,
            activation_function=head_params.activation_function,
            dense_layer=head_params.dense_layer)
    elif isinstance(head_params, PPOVHeadParameters):
        module = PPOVHead(
            agent_parameters=agent_params,
            spaces=spaces,
            network_name=network_name,
            head_type_idx=head_type_index,
            loss_weight=head_params.loss_weight,
            is_local=is_local,
            activation_function=head_params.activation_function,
            dense_layer=head_params.dense_layer)
    elif isinstance(head_params, QHeadParameters):
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


class ScaledGradHead(HybridBlock, utils.OnnxHandlerBlock):
    """
    Wrapper block for applying gradient scaling to input before feeding the head network
    """
    def __init__(self,
                 head_index: int,
                 head_type_index: int,
                 network_name: str,
                 spaces: SpacesDefinition,
                 network_is_local: bool,
                 agent_params: AgentParameters,
                 head_params: HeadParameters) -> None:
        """
        :param head_index: the head index
        :param head_type_index: the head type index (same index if head_param.num_output_head_copies>0)
        :param network_name: name of the network
        :param spaces: state and action space definitions
        :param network_is_local: whether network is local
        :param agent_params: agent parameters
        :param head_params: head parameters
        """
        super(ScaledGradHead, self).__init__()
        utils.OnnxHandlerBlock.__init__(self)

        head_params = _sanitize_activation(head_params)
        with self.name_scope():
            self.head = _get_output_head(
                head_params=head_params,
                head_idx=head_index,
                head_type_index=head_type_index,
                agent_params=agent_params,
                spaces=spaces,
                network_name=network_name,
                is_local=network_is_local)
            self.gradient_rescaler = self.params.get_constant(
                name='gradient_rescaler',
                value=np.array([float(head_params.rescale_gradient_from_head_by_factor)]))
            # self.gradient_rescaler = self.params.get(
            #     name='gradient_rescaler',
            #     shape=(1,),
            #     init=mx.init.Constant(float(head_params.rescale_gradient_from_head_by_factor)))

    def hybrid_forward(self,
                       F: ModuleType,
                       x: Union[NDArray, Symbol],
                       gradient_rescaler: Union[NDArray, Symbol]) -> Tuple[Union[NDArray, Symbol], ...]:
        """ Overrides gluon.HybridBlock.hybrid_forward
        :param nd or sym F: ndarray or symbol module
        :param x: head input
        :param gradient_rescaler: gradient rescaler for partial blocking of gradient
        :return: head output
        """
        if self._onnx:
            # ONNX doesn't support BlockGrad() operator, but it's not typically needed for
            # ONNX because mostly forward calls are performed using ONNX exported network.
            grad_scaled_x = x
        else:
            grad_scaled_x = (F.broadcast_mul((1 - gradient_rescaler), F.BlockGrad(x)) +
                             F.broadcast_mul(gradient_rescaler, x))
        out = self.head(grad_scaled_x)
        return out


class SingleModel(HybridBlock):
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
        self._input_embedders = list()  # type: List[HybridBlock]
        self._output_heads = list()  # type: List[ScaledGradHead]

        with self.name_scope():
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


class GeneralModel(HybridBlock):
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

        with self.name_scope():
            self.nets = list()
            for network_idx in range(num_networks):
                head_type_idx_start = network_idx * num_heads_per_network
                head_type_idx_end = head_type_idx_start + num_heads_per_network
                net = SingleModel(
                    head_type_idx_start=head_type_idx_start,
                    network_name=network_name,
                    network_is_local=network_is_local,
                    agent_parameters=agent_parameters,
                    in_emb_param_dict=network_parameters.input_embedders_parameters,
                    embedding_merger_type=network_parameters.embedding_merger_type,
                    middleware_param=network_parameters.middleware_parameters,
                    head_param_list=network_parameters.heads_parameters[head_type_idx_start:head_type_idx_end],
                    spaces=spaces)
                self.register_child(net)
                self.nets.append(net)

    def hybrid_forward(self, F, *inputs):
        """ Overrides gluon.HybridBlock.hybrid_forward
        :param nd or sym F: ndarray or symbol block
        :param inputs: model inputs, one for each embedder. Passed to all networks.
        :return: head outputs in a tuple
        """
        outputs = tuple()
        for net in self.nets:
            out = net(*inputs)
            outputs += out
        return outputs

    @property
    def output_heads(self) -> List[Head]:
        """ Return all heads in a single list
        Note: There is a one-to-one mapping between output_heads and losses
        :return: list of heads
        """
        return list(chain.from_iterable(net.output_heads for net in self.nets))

    def losses(self) -> List[HeadLoss]:
        """ Construct loss blocks for network training
        Note: There is a one-to-one mapping between output_heads and losses
        :return: list of loss blocks
        """
        return [h.loss() for net in self.nets for h in net.output_heads]
