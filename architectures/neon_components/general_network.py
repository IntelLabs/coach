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

from architectures.neon_components.embedders import *
from architectures.neon_components.heads import *
from architectures.neon_components.middleware import *
from architectures.neon_components.architecture import *
from configurations import InputTypes, OutputTypes, MiddlewareTypes


class GeneralNeonNetwork(NeonArchitecture):
    def __init__(self, tuning_parameters, name="", global_network=None, network_is_local=True):
        self.global_network = global_network
        self.network_is_local = network_is_local
        self.num_heads_per_network = 1 if tuning_parameters.agent.use_separate_networks_per_head else \
            len(tuning_parameters.agent.output_types)
        self.num_networks = 1 if not tuning_parameters.agent.use_separate_networks_per_head else \
            len(tuning_parameters.agent.output_types)
        self.input_embedders = []
        self.output_heads = []
        self.activation_function = self.get_activation_function(
            tuning_parameters.agent.hidden_layers_activation_function)

        NeonArchitecture.__init__(self, tuning_parameters, name, global_network, network_is_local)

    def get_activation_function(self, activation_function_string):
        activation_functions = {
            'relu': neon.Rectlin(),
            'tanh': neon.Tanh(),
            'sigmoid': neon.Logistic(),
            'elu': neon.Explin(),
            'none': None
        }
        assert activation_function_string in activation_functions.keys(), \
            "Activation function must be one of the following {}".format(activation_functions.keys())
        return activation_functions[activation_function_string]

    def get_input_embedder(self, embedder_type):
        # the observation can be either an image or a vector
        def get_observation_embedding(with_timestep=False):
            if self.input_height > 1:
                return ImageEmbedder((self.input_depth, self.input_height, self.input_width), self.batch_size,
                                     name="observation")
            else:
                return VectorEmbedder((self.input_depth, self.input_width + int(with_timestep)), self.batch_size,
                                      name="observation")

        input_mapping = {
            InputTypes.Observation: get_observation_embedding(),
            InputTypes.Measurements: VectorEmbedder(self.measurements_size, self.batch_size, name="measurements"),
            InputTypes.GoalVector: VectorEmbedder(self.measurements_size, self.batch_size, name="goal_vector"),
            InputTypes.Action: VectorEmbedder((self.num_actions,), self.batch_size, name="action"),
            InputTypes.TimedObservation: get_observation_embedding(with_timestep=True),
        }
        return input_mapping[embedder_type]

    def get_middleware_embedder(self, middleware_type):
        return {MiddlewareTypes.LSTM: None,   # LSTM over Neon is currently not supported in Coach
                MiddlewareTypes.FC: FC_Embedder}.get(middleware_type)(self.activation_function)

    def get_output_head(self, head_type, head_idx, loss_weight=1.):
        output_mapping = {
            OutputTypes.Q: QHead,
            OutputTypes.DuelingQ: DuelingQHead,
            OutputTypes.V: None, # Policy Optimization algorithms over Neon are currently not supported in Coach
            OutputTypes.Pi: None,  # Policy Optimization algorithms over Neon are currently not supported in Coach
            OutputTypes.MeasurementsPrediction: None, # DFP over Neon is currently not supported in Coach
            OutputTypes.DNDQ: None,  # NEC over Neon is currently not supported in Coach
            OutputTypes.NAF: None,  # NAF over Neon is currently not supported in Coach
            OutputTypes.PPO: None, # PPO over Neon is currently not supported in Coach
            OutputTypes.PPO_V: None  # PPO over Neon is currently not supported in Coach
        }
        return output_mapping[head_type](self.tp, head_idx, loss_weight, self.network_is_local)

    def get_model(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        :return: A model
        """
        assert len(self.tp.agent.input_types) > 0, "At least one input type should be defined"
        assert len(self.tp.agent.output_types) > 0, "At least one output type should be defined"
        assert self.tp.agent.middleware_type is not None, "Exactly one middleware type should be defined"
        assert len(self.tp.agent.loss_weights) > 0, "At least one loss weight should be defined"
        assert len(self.tp.agent.output_types) == len(self.tp.agent.loss_weights), \
            "Number of loss weights should match the number of output types"
        local_network_in_distributed_training = self.global_network is not None and self.network_is_local

        tuning_parameters.activation_function = self.activation_function
        done_creating_input_placeholders = False

        for network_idx in range(self.num_networks):
            with name_scope('network_{}'.format(network_idx)):
                ####################
                # Input Embeddings #
                ####################

                state_embedding = []
                for idx, input_type in enumerate(self.tp.agent.input_types):
                    # get the class of the input embedder
                    self.input_embedders.append(self.get_input_embedder(input_type))

                    # in the case each head uses a different network, we still reuse the input placeholders
                    prev_network_input_placeholder = self.inputs[idx] if done_creating_input_placeholders else None

                    # create the input embedder instance and store the input placeholder and the embedding
                    input_placeholder, embedding = self.input_embedders[-1](prev_network_input_placeholder)
                    if len(self.inputs) < len(self.tp.agent.input_types):
                        self.inputs.append(input_placeholder)
                    state_embedding.append(embedding)

                done_creating_input_placeholders = True

                ##############
                # Middleware #
                ##############

                state_embedding = ng.concat_along_axis(state_embedding, state_embedding[0].axes[0]) \
                    if len(state_embedding) > 1 else state_embedding[0]
                self.middleware_embedder = self.get_middleware_embedder(self.tp.agent.middleware_type)
                _, self.state_embedding = self.middleware_embedder(state_embedding)

                ################
                # Output Heads #
                ################

                for head_idx in range(self.num_heads_per_network):
                    for head_copy_idx in range(self.tp.agent.num_output_head_copies):
                        if self.tp.agent.use_separate_networks_per_head:
                            # if we use separate networks per head, then the head type corresponds top the network idx
                            head_type_idx = network_idx
                        else:
                            # if we use a single network with multiple heads, then the head type is the current head idx
                            head_type_idx = head_idx
                        self.output_heads.append(self.get_output_head(self.tp.agent.output_types[head_type_idx],
                                                                      head_copy_idx,
                                                                      self.tp.agent.loss_weights[head_type_idx]))
                        if self.network_is_local:
                            output, target_placeholder, input_placeholder = self.output_heads[-1](self.state_embedding)
                            self.targets.extend(target_placeholder)
                        else:
                            output, input_placeholder = self.output_heads[-1](self.state_embedding)

                        self.outputs.extend(output)
                        self.inputs.extend(input_placeholder)

        # Losses
        self.losses = []
        for output_head in self.output_heads:
            self.losses += output_head.loss
        self.total_loss = sum(self.losses)

        # Learning rate
        if self.tp.learning_rate_decay_rate != 0:
            raise Exception("learning rate decay is not supported in neon")

        # Optimizer
        if local_network_in_distributed_training and \
                hasattr(self.tp.agent, "shared_optimizer") and self.tp.agent.shared_optimizer:
            # distributed training and this is the local network instantiation
            self.optimizer = self.global_network.optimizer
        else:
            if tuning_parameters.agent.optimizer_type == 'Adam':
                self.optimizer = neon.Adam(
                    learning_rate=tuning_parameters.learning_rate,
                    gradient_clip_norm=tuning_parameters.clip_gradients
                )
            elif tuning_parameters.agent.optimizer_type == 'RMSProp':
                self.optimizer = neon.RMSProp(
                    learning_rate=tuning_parameters.learning_rate,
                    gradient_clip_norm=tuning_parameters.clip_gradients,
                    decay_rate=0.9,
                    epsilon=0.01
                )
            elif tuning_parameters.agent.optimizer_type == 'LBFGS':
                raise Exception("LBFGS optimizer is not supported in neon")
            else:
                raise Exception("{} is not a valid optimizer type".format(tuning_parameters.agent.optimizer_type))
