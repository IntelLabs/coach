
import copy
import tensorflow as tf
from tensorflow import keras
from typing import List
from itertools import chain
import numpy as np

from types import ModuleType
from rl_coach.architectures.tensorflow_components.embedders import ImageEmbedder, TensorEmbedder, VectorEmbedder
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters, LSTMMiddlewareParameters
from rl_coach.architectures.tensorflow_components.middlewares import FCMiddleware, LSTMMiddleware
#from rl_coach.architectures.tensorflow_components.heads import Head, QHead
from rl_coach.architectures.tensorflow_components.heads import Head, PPOHead, PPOVHead, VHead, QHead


#from rl_coach.architectures.head_parameters import QHeadParameters
from rl_coach.architectures.head_parameters import HeadParameters, PPOHeadParameters
from rl_coach.architectures.head_parameters import PPOVHeadParameters, VHeadParameters, QHeadParameters

from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.middleware_parameters import MiddlewareParameters
from rl_coach.base_parameters import AgentParameters, EmbeddingMergerType
from rl_coach.base_parameters import NetworkParameters
from rl_coach.spaces import SpacesDefinition, PlanarMapsObservationSpace, TensorObservationSpace


class SingleDnnModel(keras.Model):
    """
    Block that connects a single embedder, with middleware and one to multiple heads
    """
    def __init__(self,
                 network_is_local: bool,
                 network_name: str,
                 agent_parameters: AgentParameters,
                 input_embedders_parameters: {str: InputEmbedderParameters},
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
        :param input_embedders_parameters: dictionary of embedder name to embedding parameters
        :param embedding_merger_type: type of merging output of embedders: concatenate or sum
        :param middleware_param: middleware parameters
        :param head_param_list: list of head parameters, one per head type
        :param head_type_idx_start: start index for head type index counting
        :param spaces: state and action space definition
        """
        super(SingleDnnModel, self).__init__(*args, **kwargs)

        self._embedding_merger_type = embedding_merger_type
        self._input_embedders = []
        self._output_heads = list()



        for input_name in sorted(input_embedders_parameters):
            input_type = input_embedders_parameters[input_name]
            input_embedder = self._get_input_embedder(spaces, input_name, input_type)
            self._input_embedders.append(input_embedder)



        self.middleware = self._get_middleware(middleware_param)

        for i, head_param in enumerate(head_param_list):
            for head_copy_idx in range(head_param.num_output_head_copies):
                # create output head and add it to the output heads list
                head_idx = (head_type_idx_start + i) * head_param.num_output_head_copies + head_copy_idx
                output_head = self._get_output_head(
                    head_idx=head_idx,
                    head_type_index=head_type_idx_start + i,
                    network_name=network_name,
                    spaces=spaces,
                    is_local=network_is_local,
                    agent_params=agent_parameters,
                    head_params=head_param)

                self._output_heads.append(output_head)

                self.gradient_rescaler = 1





    def call(self, inputs, **kwargs):
        """ Overrides tf.keras.call
        :param inputs: model inputs, one for each embedder
        :return: head outputs in a tuple
        """
        # Input Embeddings
        state_embedding = list()
        for input, embedder in zip(inputs, self._input_embedders):
            state_embedding.append(embedder(input))


        # Merger
        if len(state_embedding) == 1:
            # TODO: change to squeeze
            state_embedding = state_embedding[0]
        else:
            if self._embedding_merger_type == EmbeddingMergerType.Concat:
                state_embedding = tf.keras.layers.Concatenate()(state_embedding)
            elif self._embedding_merger_type == EmbeddingMergerType.Sum:
                state_embedding = tf.keras.layers.Add()(state_embedding)

        # Middleware
        state_embedding = self.middleware(state_embedding)

        head_input = (1 - self.gradient_rescaler) * tf.stop_gradient(state_embedding) + self.gradient_rescaler * state_embedding

        # Head
        outputs = tuple()
        for head in self._output_heads:
            out = head(head_input)
            if not isinstance(out, tuple):
                out = (out,)
            outputs += out

        return outputs

    def _get_input_embedder(self, spaces: SpacesDefinition,
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
            #module = ImageEmbedder(embedder_params)
            module = ImageEmbedder(input_size=allowed_inputs[input_name].shape,
                                   activation_function=embedder_params.activation_function,
                                   scheme=embedder_params.scheme,
                                   batchnorm=embedder_params.batchnorm,
                                   dropout_rate=embedder_params.dropout_rate,
                                   name=embedder_params.name,
                                   input_rescaling=embedder_params.input_rescaling[type],
                                   input_offset=embedder_params.input_offset[type],
                                   input_clipping=embedder_params.input_clipping,
                                   is_training=embedder_params.is_training)

        elif type == 'tensor':
            module = TensorEmbedder(embedder_params)
        else:
            raise KeyError('Unsupported embedder type: {}'.format(type))
        return module

    def _get_middleware(self, middleware_params: MiddlewareParameters) -> ModuleType:
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

    def _get_output_head(self,
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

        elif isinstance(head_params, PPOHeadParameters):
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

        else:
            raise KeyError('Unsupported head type: {}'.format(type(head_params)))

        return module

    @property
    def input_embedders(self):
        """
        :return: list of input embedders
        """
        return self._input_embedders

    # @property
    # def heads(self):
    #     """
    #     :return: list of output heads
    #     """
    #     #return [h.head for h in self._output_heads]
    #     return self._output_heads

    @property
    def output_heads(self) -> List[Head]:
        """
        :return: list of output heads
        """
        return self._output_heads
        #return [h.head for h in self._output_heads]


class DnnModel(keras.Model):
    """
    Block that creates two single models. One for the actor and one for the critic
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
        super(DnnModel, self).__init__(*args, **kwargs)

        self.nets = list()
        for network_idx in range(num_networks):
            head_type_idx_start = network_idx * num_heads_per_network
            head_type_idx_end = head_type_idx_start + num_heads_per_network
            net = SingleDnnModel(
                head_type_idx_start=head_type_idx_start,
                network_name=network_name,
                network_is_local=network_is_local,
                agent_parameters=agent_parameters,
                input_embedders_parameters=network_parameters.input_embedders_parameters,
                embedding_merger_type=network_parameters.embedding_merger_type,
                middleware_param=network_parameters.middleware_parameters,
                head_param_list=network_parameters.heads_parameters[head_type_idx_start:head_type_idx_end],
                spaces=spaces)
            self.nets.append(net)

        self._input_shapes = self._get_input_shapes(spaces, self.nets[0].input_embedders)


    def _get_input_shapes(self, spaces, input_embedders) -> List[List[int]]:
        """
        Create a list of input array shapes
        :return: type of input shapes
        """
        allowed_inputs = copy.copy(spaces.state.sub_spaces)
        allowed_inputs["action"] = copy.copy(spaces.action)
        allowed_inputs["goal"] = copy.copy(spaces.goal)
        return list([1] + allowed_inputs[emb.embedder_name].shape.tolist() for emb in input_embedders)


    @property
    def dummy_model_inputs(self):
        """
        Creates a tuple of input arrays with correct shapes that can be used for shape inference
        of the model weights and for printing the summary
        :return: tuple of inputs for model forward pass
        """
        input_shapes = self._input_shapes
        # TODO make this the same type as the actual input
        inputs = tuple(np.zeros(tuple(shape)) for shape in input_shapes)
        return inputs

    def call(self, inputs, **kwargs):
        """ Overrides tf.keras.call
        :param inputs: model inputs, one for each embedder. Passed to all networks.
        :return: head outputs in a tuple
        """
        outputs = tuple()
        for net in self.nets:
            out = net(inputs)
            num_outputs = len(out)

            if net.output_heads[0]._num_outputs is None:
                net.output_heads[0]._num_outputs = num_outputs
            else:
                assert net.output_heads[0]._num_outputs == num_outputs, 'Number of outputs cannot change ({} != {})'.format(
                    net.output_heads[0]._num_outputs, num_outputs)
            outputs += out
            #outputs.append(out)
        return outputs

    @property
    def output_heads(self) -> List[Head]:
        """ Return all heads in a single list
        Note: There is a one-to-one mapping between output_heads and losses
        :return: list of heads
        """
        return list(chain.from_iterable(net.output_heads for net in self.nets))



