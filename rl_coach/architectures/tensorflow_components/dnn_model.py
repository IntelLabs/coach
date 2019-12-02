
import copy
import tensorflow as tf
from tensorflow import keras
from typing import List
from itertools import chain
import numpy as np
from tensorflow.keras.layers import Input, Dense


from types import ModuleType
from rl_coach.architectures.tensorflow_components.embedders import ImageEmbedder, TensorEmbedder, VectorEmbedder
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters, LSTMMiddlewareParameters
from rl_coach.architectures.tensorflow_components.middlewares import FCMiddleware, LSTMMiddleware
#from rl_coach.architectures.tensorflow_components.heads import Head, QHead
from rl_coach.architectures.tensorflow_components.heads import Head#, PPOHead, PPOVHead, VHead, QHead
from rl_coach.architectures.tensorflow_components.heads.ppo_head import continuous_ppo_head
from rl_coach.architectures.tensorflow_components.heads.v_head import value_head

#from rl_coach.architectures.head_parameters import QHeadParameters
from rl_coach.architectures.head_parameters import HeadParameters, PPOHeadParameters
from rl_coach.architectures.head_parameters import PPOVHeadParameters, VHeadParameters, QHeadParameters

from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.middleware_parameters import MiddlewareParameters
from rl_coach.base_parameters import AgentParameters, EmbeddingMergerType
from rl_coach.base_parameters import NetworkParameters
from rl_coach.spaces import SpacesDefinition, PlanarMapsObservationSpace, TensorObservationSpace


def _get_input_embedder(name_prefix: str,
                        spaces: SpacesDefinition,
                        input_name: str,
                        embedder_params: InputEmbedderParameters) -> ModuleType:
    """
    Given an input embedder parameters class, creates the input embedder and returns it
    :param input_name: the name of the input to the embedder (used for retrieving the shape). The input should
                       be a value within the state or the action.
    :param embedder_params: the parameters of the class of the embedder
    :return: the embedder instance
    """
    embedder_params = copy.copy(embedder_params)

    embedder_params.name = name_prefix + '_input_embedder'

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
                                #scheme=[Dense(64)],
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



def _get_middleware(middleware_params: MiddlewareParameters) -> ModuleType:
    """
    Given a middleware type, creates the middleware and returns it
    :param middleware_params: the paramaeters of the middleware class
    :return: the middleware instance
    """
    if isinstance(middleware_params, FCMiddlewareParameters):
        module = FCMiddleware(activation_function=middleware_params.activation_function,
                              scheme=middleware_params.scheme,
                              #scheme=[Dense(64)],
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
    :return: head name and head block
    """

    if isinstance(head_params, QHeadParameters):
        head_input_dim = 512 # middleware output dim hard coded, because scheme is hard coded
        head_output_dim = len(spaces.action.actions)
        module = value_head(head_input_dim, head_output_dim)

        # module = QHead(
        #     agent_parameters=agent_params,
        #     spaces=spaces,
        #     network_name=network_name,
        #     head_type_idx=head_type_index,
        #     loss_weight=head_params.loss_weight,
        #     is_local=is_local,
        #     activation_function=head_params.activation_function,
        #     dense_layer=head_params.dense_layer)

    elif isinstance(head_params, PPOHeadParameters):
        head_input_dim = 64  # middleware output dim hard coded, because scheme is hard coded
        head_output_dim = spaces.action.shape[0]
        module = continuous_ppo_head(head_input_dim, head_output_dim)

        # module = PPOHead(
        #     agent_parameters=agent_params,
        #     spaces=spaces,
        #     network_name=network_name,
        #     head_type_idx=head_type_index,
        #     loss_weight=head_params.loss_weight,
        #     is_local=is_local,
        #     activation_function=head_params.activation_function,
        #     dense_layer=head_params.dense_layer)

    elif isinstance(head_params, VHeadParameters):
        head_input_dim = 64  # middleware output dim hard coded, because scheme is hard coded
        head_output_dim = 1
        module = value_head(head_input_dim, head_output_dim)

        #
        # module = VHead(
        #     agent_parameters=agent_params,
        #     spaces=spaces,
        #     network_name=network_name,
        #     head_type_idx=head_type_index,
        #     loss_weight=head_params.loss_weight,
        #     is_local=is_local,
        #     activation_function=head_params.activation_function,
        #     dense_layer=head_params.dense_layer)


    else:
        raise KeyError('Unsupported head type: {}'.format(type(head_params)))

    return module


def create_single_network(inputs_shapes,
                          name: str,
                          network_is_local: bool,
                          head_type_idx_start: int,
                          agent_parameters: AgentParameters,
                          input_embedders_parameters: {str: InputEmbedderParameters},
                          embedding_merger_type: EmbeddingMergerType,
                          middleware_param: MiddlewareParameters,
                          head_param_list: [HeadParameters],
                          spaces: SpacesDefinition):
    """
    :param network_is_local: True if network is local
    :param name: name of the network
    :param agent_parameters: agent parameters
    :param input_embedders_parameters: dictionary of embedder name to embedding parameters
    :param embedding_merger_type: type of merging output of embedders: concatenate or sum
    :param middleware_param: middleware parameters
    :param head_param_list: list of head parameters, one per head type
    :param head_type_idx_start: start index for head type index counting
    :param spaces: state and action space definition
    """
    name = name + '_' + head_param_list[0].name.replace('head_params', '_network')
    inputs = list(map(lambda x: Input(name=name + '_input', shape=x), inputs_shapes))
    # Get list of input embedders
    embedders = [_get_input_embedder(name, spaces, k, v) for k, v in input_embedders_parameters.items()]
    # Apply each embbeder on its corresponding input
    state_embeddings = [embedder(input_t) for embedder, input_t in zip(embedders, inputs)]


    # Merge embedders outputs
    if len(state_embeddings) == 1:
        # TODO: change to squeeze
        state_embeddings = state_embeddings[0]
    else:
        if embedding_merger_type == EmbeddingMergerType.Concat:
            state_embeddings = tf.keras.layers.Concatenate()(state_embeddings)
        elif embedding_merger_type == EmbeddingMergerType.Sum:
            state_embeddings = tf.keras.layers.Add()(state_embeddings)

    middleware_output = _get_middleware(middleware_param)(state_embeddings)

    heads_outputs = list()

    for i, head_param in enumerate(head_param_list):
        for head_copy_idx in range(head_param.num_output_head_copies):
            # create output head and add it to the output heads list
            head_idx = (head_type_idx_start + i) * head_param.num_output_head_copies + head_copy_idx
            network_head = _get_output_head(
                head_idx=head_idx,
                head_type_index=head_type_idx_start + i,
                network_name=name,
                spaces=spaces,
                is_local=network_is_local,
                agent_params=agent_parameters,
                head_params=head_param)

            heads_outputs.append(network_head(middleware_output))
            gradient_rescaler = 1
            #return gradient_rescaler

    name = name + '_' + head_param_list[0].name.replace('head_params', '_network')
    model = keras.Model(name=name, inputs=inputs, outputs=heads_outputs)
    return model


def create_full_model(num_networks: int,
                      num_heads_per_network: int,
                      network_is_local: bool,
                      network_name: str,
                      agent_parameters: AgentParameters,
                      network_parameters: NetworkParameters,
                      spaces: SpacesDefinition):
    """
    function that creates two single models. One for the actor and one for the critic
    :param num_networks: number of networks to create
    :param num_heads_per_network: number of heads per network to create
    :param network_is_local: True if network is local
    :param network_name: name of the network
    :param agent_parameters: agent parameters
    :param network_parameters: network parameters
    :param spaces: state and action space definitions
    """

    input_emmbeders_types = network_parameters.input_embedders_parameters.keys()
    input_shapes = get_input_shapes(spaces, input_emmbeders_types)
    inputs = list(map(lambda x: Input(name=network_name + '_input', shape=x), input_shapes))

    outputs = list()
    networks = {}
    for network_idx in range(num_networks):
        #network_parameters = copy.deepcopy(network_params)
        head_type_idx_start = network_idx * num_heads_per_network
        head_type_idx_end = head_type_idx_start + num_heads_per_network
        networks[network_idx] = create_single_network(inputs_shapes=input_shapes,
                                    name=network_name,
                                    network_is_local=network_is_local,
                                    head_type_idx_start=head_type_idx_start,
                                    agent_parameters=agent_parameters,
                                    input_embedders_parameters=network_parameters.input_embedders_parameters,
                                    embedding_merger_type=network_parameters.embedding_merger_type,
                                    middleware_param=network_parameters.middleware_parameters,
                                    head_param_list=network_parameters.heads_parameters[head_type_idx_start:head_type_idx_end],
                                    spaces=spaces)

        outputs.append(networks[network_idx](inputs))

    model = keras.Model(name=network_name + '_full_model', inputs=inputs, outputs=outputs)
    # Dummy batch size 1, therefore [1] + shape
    dummy_inputs = tuple(np.zeros(tuple([1] + shape)) for shape in input_shapes)
    model(dummy_inputs)
    return model


def get_input_shapes(spaces, input_emmbeders_types) -> List[List[int]]:
    """
    Create a list of input array shapes
    :return: type of input shapes
    """
    allowed_inputs = copy.copy(spaces.state.sub_spaces)
    allowed_inputs["action"] = copy.copy(spaces.action)
    allowed_inputs["goal"] = copy.copy(spaces.goal)
    return list(allowed_inputs[embedder_type].shape.tolist() for embedder_type in input_emmbeders_types)


def squeeze_model_outputs(model_outputs):
    if len(model_outputs) == 1:
        return model_outputs
    else:
        return list(map(lambda output: output[0], model_outputs))

