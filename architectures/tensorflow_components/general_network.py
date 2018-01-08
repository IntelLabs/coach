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

from architectures.tensorflow_components.embedders import *
from architectures.tensorflow_components.heads import *
from architectures.tensorflow_components.middleware import *
from architectures.tensorflow_components.architecture import *
from configurations import InputTypes, OutputTypes, MiddlewareTypes


class GeneralTensorFlowNetwork(TensorFlowArchitecture):
    """
    A generalized version of all possible networks implemented using tensorflow.
    """
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

        TensorFlowArchitecture.__init__(self, tuning_parameters, name, global_network, network_is_local)

    def get_activation_function(self, activation_function_string):
        activation_functions = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'elu': tf.nn.elu,
            'none': None
        }
        assert activation_function_string in activation_functions.keys(), \
            "Activation function must be one of the following {}".format(activation_functions.keys())
        return activation_functions[activation_function_string]

    def get_input_embedder(self, embedder_type):
        # the observation can be either an image or a vector
        def get_observation_embedding(with_timestep=False):
            if self.input_height > 1:
                return ImageEmbedder((self.input_height, self.input_width, self.input_depth), name="observation")
            else:
                return VectorEmbedder((self.input_width + int(with_timestep), self.input_depth), name="observation")

        input_mapping = {
            InputTypes.Observation: get_observation_embedding(),
            InputTypes.Measurements: VectorEmbedder(self.measurements_size, name="measurements"),
            InputTypes.GoalVector: VectorEmbedder(self.measurements_size, name="goal_vector"),
            InputTypes.Action: VectorEmbedder((self.num_actions,), name="action"),
            InputTypes.TimedObservation: get_observation_embedding(with_timestep=True),
        }
        return input_mapping[embedder_type]

    def get_middleware_embedder(self, middleware_type):
        return {MiddlewareTypes.LSTM: LSTM_Embedder,
                MiddlewareTypes.FC: FC_Embedder}.get(middleware_type)(self.activation_function)

    def get_output_head(self, head_type, head_idx, loss_weight=1.):
        output_mapping = {
            OutputTypes.Q: QHead,
            OutputTypes.DuelingQ: DuelingQHead,
            OutputTypes.V: VHead,
            OutputTypes.Pi: PolicyHead,
            OutputTypes.MeasurementsPrediction: MeasurementsPredictionHead,
            OutputTypes.DNDQ: DNDQHead,
            OutputTypes.NAF: NAFHead,
            OutputTypes.PPO: PPOHead,
            OutputTypes.PPO_V: PPOVHead,
            OutputTypes.CategoricalQ: CategoricalQHead,
            OutputTypes.QuantileRegressionQ: QuantileRegressionQHead
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

        for network_idx in range(self.num_networks):
            with tf.variable_scope('network_{}'.format(network_idx)):
                ####################
                # Input Embeddings #
                ####################

                state_embedding = []
                for idx, input_type in enumerate(self.tp.agent.input_types):
                    # get the class of the input embedder
                    input_embedder = self.get_input_embedder(input_type)
                    self.input_embedders.append(input_embedder)

                    # input placeholders are reused between networks. on the first network, store the placeholders
                    # generated by the input_embedders in self.inputs. on the rest of the networks, pass
                    # the existing input_placeholders into the input_embedders.
                    if network_idx == 0:
                        input_placeholder, embedding = input_embedder()
                        self.inputs.append(input_placeholder)
                    else:
                        input_placeholder, embedding = input_embedder(self.inputs[idx])

                    state_embedding.append(embedding)

                ##############
                # Middleware #
                ##############

                state_embedding = tf.concat(state_embedding, axis=-1) if len(state_embedding) > 1 else state_embedding[0]
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

                        if self.tp.agent.stop_gradients_from_head[head_idx]:
                            head_input = tf.stop_gradient(self.state_embedding)
                        else:
                            head_input = self.state_embedding

                        # build the head
                        if self.network_is_local:
                            output, target_placeholder, input_placeholder = self.output_heads[-1](head_input)
                            self.targets.extend(target_placeholder)
                        else:
                            output, input_placeholder = self.output_heads[-1](head_input)

                        self.outputs.extend(output)
                        self.inputs.extend(input_placeholder)

        # Losses
        self.losses = tf.losses.get_losses(self.name)
        self.losses += tf.losses.get_regularization_losses(self.name)
        self.total_loss = tf.losses.compute_weighted_loss(self.losses, scope=self.name)

        # Learning rate
        if self.tp.learning_rate_decay_rate != 0:
            self.tp.learning_rate = tf.train.exponential_decay(
                self.tp.learning_rate, self.global_step, decay_steps=self.tp.learning_rate_decay_steps,
                decay_rate=self.tp.learning_rate_decay_rate, staircase=True)

        # Optimizer
        if local_network_in_distributed_training and \
                hasattr(self.tp.agent, "shared_optimizer") and self.tp.agent.shared_optimizer:
            # distributed training and this is the local network instantiation
            self.optimizer = self.global_network.optimizer
        else:
            if tuning_parameters.agent.optimizer_type == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=tuning_parameters.learning_rate)
            elif tuning_parameters.agent.optimizer_type == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.tp.learning_rate, decay=0.9, epsilon=0.01)
            elif tuning_parameters.agent.optimizer_type == 'LBFGS':
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B',
                                                                        options={'maxiter': 25})
            else:
                raise Exception("{} is not a valid optimizer type".format(tuning_parameters.agent.optimizer_type))
