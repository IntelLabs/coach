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

                #self._output_heads.append(output_head)
                self._output_heads = output_head

    def call(self, inputs, **kwargs):
        """ Overrides tf.keras.call
        :param inputs: model inputs, one for each embedder
        :return: head outputs in a tuple
        """
        # Input Embeddings
        #state_embedding = map()
        state_embedding = [ ]
        for input, embedder in zip(inputs, self._input_embedders):
            state_embedding.append(embedder(input))

        # Merger
        if len(state_embedding) == 1:
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
        # outputs = tuple()
        # for head in self._output_heads:
        #     out = head(state_embedding)
        #     if not isinstance(out, tuple):
        #         out = (out,)
        #     outputs += out

        # Dan for debug
        outputs = self._output_head(state_embedding)
        return outputs



    def call(self, inputs):

        embedded_inputs = []
        for embedder in self.input_embedders:
            embedded_inputs.append(embedder(inputs))


        if len(embedded_inputs) == 1:
            state_embedding = embedded_inputs[0]
        else:
            if self.network_parameters.embedding_merger_type == EmbeddingMergerType.Concat:
                state_embedding = tf.keras.layers.Concatenate()(embedded_inputs)
            elif self.network_parameters.embedding_merger_type == EmbeddingMergerType.Sum:
                state_embedding = tf.keras.layers.Add()(embedded_inputs)

        middleware_out = self.middleware(state_embedding)


        output = self.out_head(middleware_out)
        return output










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
        outputs = self.single_model(inputs)
        # outputs = tuple()
        # for net in self.nets:
        #     out = net(inputs)
        #     outputs += out


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










class GeneralModel2(keras.Model):

    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str,
                 global_network=None, network_is_local: bool=True, network_is_trainable: bool=False, **kwargs):
        super().__init__(**kwargs)
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """

        ##Dan
        self.ap = AgentParameters
        self.spaces = spaces
        # size of the action space for now, should be updated for actor critic
        self.output_dim = 2
        ##


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

        self.is_training = None

        for input_name in sorted(self.network_parameters.input_embedders_parameters):
            # input_type = self.network_parameters.input_embedders_parameters[input_name]
            # Dan- changed input_type to more informative name
            embbeder_parameters = self.network_parameters.input_embedders_parameters[input_name]
            # Creates input embedder object (calls init)
            input_embedder = _get_input_embedder(spaces, input_name, embbeder_parameters)
            #input_embedder = self.get_input_embedder(input_name, embbeder_parameters)
            self.input_embedders.append(input_embedder)

        #self.middleware = self.get_middleware(self.network_parameters.middleware_parameters)
        self.middleware = _get_middleware(self.network_parameters.middleware_parameters)




        # #########
        head_count = 0

        # for head_idx in range(self.num_heads_per_network):
        #     #if self.network_parameters.use_separate_networks_per_head:
        #     #     # if we use separate networks per head, then the head type corresponds to the network idx
        #     #     head_type_idx = 0#network_idx
        #     #     head_count = 0#network_idx
        #     # else:
        #     #     # if we use a single network with multiple embedders, then the head type is the current head idx
        #     #     head_type_idx = head_idx
        #     head_type_idx = head_idx
        #     head_params = self.network_parameters.heads_parameters[head_type_idx]
        #
        #     for head_copy_idx in range(head_params.num_output_head_copies):
        #         # create output head and add it to the output heads list
        #         self.output_heads.append(
        #             self.get_output_head(head_params,
        #                                  head_idx * head_params.num_output_head_copies + head_copy_idx)
        #         )

        # TODO: starting with q learning, will handle actor critic multiple outputs later
        head_params = self.network_parameters.heads_parameters[0]

        #self.out_head = self.get_output_head(head_params, 0)
        #self.out_head = _get_output_head(head_params)
        self.out_head = _get_output_head(
            head_idx=0,
            head_type_index=0,
            network_name=self.name,
            spaces=spaces,
            is_local=network_is_local,
            agent_params=agent_parameters,
            head_params=head_params)


    def call(self, inputs):

        embedded_inputs = []
        outputs = []
        for embedder in self.input_embedders:
            embedded_inputs.append(embedder(inputs))

        ##########
        # Merger #
        ##########
        if len(embedded_inputs) == 1:
            state_embedding = embedded_inputs[0]
        else:
            if self.network_parameters.embedding_merger_type == EmbeddingMergerType.Concat:
                state_embedding = tf.keras.layers.Concatenate()(embedded_inputs)
            elif self.network_parameters.embedding_merger_type == EmbeddingMergerType.Sum:
                state_embedding = tf.keras.layers.Add()(embedded_inputs)

        middleware_out = self.middleware(state_embedding)

        # for head in self.output_heads:
        #     outputs.extend(head(middleware_out))
        # return outputs
        output = self.out_head(middleware_out)
        return output
        # for head in self.output_heads:
        #     outputs.extend(head(middleware_out))
        # return outputs

        #return middleware_out


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)





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



    def get_output_head(self, head_params: HeadParameters, head_idx: int):
        """
        Given a head type, creates the head and returns it
        :param head_params: the parameters of the head to create
        :param head_idx: the head index
        :return: the head
        """
        mod_name = head_params.parameterized_class_name
        head_path = head_params.path
        head_params_copy = copy.copy(head_params)
        head_params_copy.activation_function = utils.get_activation_function(head_params_copy.activation_function)
        head_params_copy.is_training = self.is_training
        return dynamic_import_and_instantiate_module_from_params(head_params_copy, path=head_path, extra_kwargs={
            'agent_parameters': self.ap, 'spaces': self.spaces, 'network_name': self.network_wrapper_name,
            'head_idx': head_idx, 'is_local': self.network_is_local})

    def get_model(self) -> List:
        # validate the configuration
        if len(self.network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(self.network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if self.network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        # ops for defining the training / testing phase
        # Dan manual fix
        # self.is_training = tf.Variable(False, trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        # self.is_training_placeholder = tf.compat.v1.placeholder("bool")

        self.is_training = tf.compat.v1.Variable(False, trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
        self.is_training_placeholder = self.is_training.numpy()
        self.assign_is_training = tf.compat.v1.assign(self.is_training, self.is_training_placeholder)

        for network_idx in range(self.num_networks):
            with tf.compat.v1.variable_scope('network_{}'.format(network_idx)):

                ####################
                # Input Embeddings #
                ####################

                state_embedding = []
                for input_name in sorted(self.network_parameters.input_embedders_parameters):
                    #input_type = self.network_parameters.input_embedders_parameters[input_name]
                    # Dan- changed input_type to more informative name
                    embbeder_parameters = self.network_parameters.input_embedders_parameters[input_name]
                    # Creates input embedder object (calls init)
                    input_embedder = self.get_input_embedder(input_name, embbeder_parameters)

                    self.input_embedders.append(input_embedder)



                    # Dan manual fix no need for placehoders in tf 2
                    # # input placeholders are reused between networks. on the first network, store the placeholders
                    # # generated by the input_embedders in self.inputs. on the rest of the networks, pass
                    # # the existing input_placeholders into the input_embedders.
                    # if network_idx == 0:
                    #     input_placeholder, embedding = input_embedder()
                    #     self.inputs[input_name] = input_placeholder
                    # else:
                    #     input_placeholder, embedding = input_embedder(self.inputs[input_name])
                    #
                    #state_embedding.append(embedding)

                ##########
                # Merger #
                ##########

                if len(self.input_embedders) == 1:
                    state_embedding = self.input_embedders[0]
                else:
                    if self.network_parameters.embedding_merger_type == EmbeddingMergerType.Concat:
                        state_embedding = tf.keras.layers.Concatenate(self.input_embedders, axis=-1, name="merger")
                        # Dan manual fix
                        #state_embedding = tf.concat(state_embedding, axis=-1, name="merger")
                    elif self.network_parameters.embedding_merger_type == EmbeddingMergerType.Sum:
                        state_embedding = tf.add_n(self.input_embedders, name="merger")
                        # state_embedding = tf.add_n(state_embedding, name="merger")

                ##############
                # Middleware #
                ##############

                self.middleware = self.get_middleware(self.network_parameters.middleware_parameters)
                #this should be in call
                #_, self.state_embedding = self.middleware(state_embedding)

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
                            tf.compat.v1.get_variable('gradients_from_head_{}-{}_rescalers'.format(head_idx, head_copy_idx),
                                            initializer=float(head_params.rescale_gradient_from_head_by_factor),
                                            dtype=tf.float32,
                                            use_resource=False))
                                            #  Dan manual fix: use_resource=False

                        self.gradients_from_head_rescalers_placeholders.append(
                            tf.compat.v1.placeholder('float',
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

        # model weights
        if not self.distributed_training or self.network_is_global:
            self.weights = [var for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.full_name) if
                            'global_step' not in var.name]
        else:
            self.weights = [var for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.full_name)]

        # Losses
        self.losses = tf.compat.v1.losses.get_losses(self.full_name)

        # L2 regularization
        if self.network_parameters.l2_regularization != 0:
            self.l2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.weights]) \
                                     * self.network_parameters.l2_regularization
            self.losses += self.l2_regularization

        self.total_loss = tf.reduce_sum(input_tensor=self.losses)
        # tf.summary.scalar('total_loss', self.total_loss)



        return self.weights

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
