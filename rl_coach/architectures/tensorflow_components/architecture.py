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
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Distribution

from rl_coach.architectures.architecture import Architecture

from rl_coach.base_parameters import AgentParameters
from rl_coach.logger import screen
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list, squeeze_list
from rl_coach.architectures.tensorflow_components import utils
from rl_coach.architectures.tensorflow_components.dnn_model import squeeze_model_outputs
from rl_coach.architectures.tensorflow_components.heads import Head
#from rl_coach.architectures.tensorflow_components.losses import HeadLoss
from tensorflow.keras.losses import Loss
from rl_coach.core_types import GradientClippingMethod
from rl_coach.architectures.tensorflow_components.savers import TfSaver
#from memory_profiler import profile
from rl_coach.architectures.tensorflow_components.losses.head_loss import LOSS_OUT_TYPE_LOSS, LOSS_OUT_TYPE_REGULARIZATION
from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp
from rl_coach.architectures.tensorflow_components.losses.ppo_loss import PPOLoss
from rl_coach.architectures.tensorflow_components.losses.q_loss import QLoss, q_loss_f
from rl_coach.architectures.tensorflow_components.losses.v_loss import v_loss_f
from rl_coach.architectures.tensorflow_components.losses.ppo_loss import ppo_loss_f

class TensorFlowArchitecture(Architecture):
    def __init__(self, agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 name: str = "",
                 global_network=None,
                 network_is_local: bool=True,
                 network_is_trainable: bool=False):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """
        super().__init__(agent_parameters, spaces, name)
        self.middleware = None
        self.network_is_local = network_is_local
        self.global_network = global_network
        if not self.network_parameters.tensorflow_support:
            raise ValueError('TensorFlow is not supported for this agent')
        self.losses = []  # type: List[Loss]
        self.shared_accumulated_gradients = []
        self.curr_rnn_c_in = None
        self.curr_rnn_h_in = None
        self.gradients_wrt_inputs = []
        self.train_writer = None
        self.accumulated_gradients = None
        self.network_is_trainable = network_is_trainable
        self.is_training = False
        self.model = None  # type: GeneralModel

        self.is_chief = self.ap.task_parameters.task_index == 0
        self.network_is_global = not self.network_is_local and global_network is None
        self.distributed_training = self.network_is_global or self.network_is_local and global_network is not None

        self.optimizer_type = self.network_parameters.optimizer_type
        self.emmbeding_types = list(self.network_parameters.input_embedders_parameters.keys())
        if self.ap.task_parameters.seed is not None:
            tf.compat.v1.set_random_seed(self.ap.task_parameters.seed)

        # Call to child class to create the model
        self.construct_model()

        self.trainer = None

    def __str__(self):
        return self.model.summary(self._dummy_model_inputs())



    def construct_model(self) -> None:
        """
        Construct network model. Implemented by child class.
        """
        print('Construct is empty for now and is called from class constructor')

    def set_session(self, sess) -> None:
        """
        Initializes the model parameters and creates the model trainer.
        NOTEL Session for TF2 backend must be None.
        :param sess: must be None
        """
        assert sess is None

    def reset_accumulated_gradients(self) -> None:
        """
        Reset the gradients accumulation
        """

        if self.accumulated_gradients is None:
            self.accumulated_gradients = self.model.get_weights().copy()

        self.accumulated_gradients = list(map(lambda grad: grad * 0, self.accumulated_gradients))

    #@profile
    def accumulate_gradients(self,
                             inputs: Dict[str, np.ndarray],
                             targets: List[np.ndarray],
                             additional_fetches: List[Tuple[int, str]] = None,
                             importance_weights: np.ndarray = None,
                             no_accumulation: bool = False) -> Tuple[float, List[float], float, list]:
        """
        Runs a forward & backward pass, clips gradients if needed and accumulates them into the accumulation
        :param inputs: environment states (observation, etc.) as well extra inputs required by loss. Shape of ndarray
            is (batch_size, observation_space_size) or (batch_size, observation_space_size, stack_size)
        :param targets: targets required by  loss (e.g. sum of discounted rewards)
        :param additional_fetches: additional fetches to calculate and return. Each fetch is specified as (int, str)
            tuple of head-type-index and fetch-name. The tuple is obtained from each head.
        :param importance_weights: ndarray of shape (batch_size,) to multiply with batch loss.
        :param no_accumulation: if True, set gradient values to the new gradients, otherwise sum with previously
            calculated gradients
        :return: tuple of total_loss, losses, norm_unclipped_grads, fetched_tensors
            total_loss (float): sum of all head losses
            losses (list of float): list of all losses. The order is list of target losses followed by list of
                regularization losses. The specifics of losses is dependant on the network parameters
                (number of heads, etc.)
            norm_unclippsed_grads (float): global norm of all gradients before any gradient clipping is applied
            fetched_tensors: all values for additional_fetches
        """

        assert self.middleware.__class__.__name__ != 'LSTMMiddleware', "LSTM middleware not supported"

        if self.accumulated_gradients is None:
            self.reset_accumulated_gradients()

        heads_indices = list(range(len(self.model.outputs)))
        model_inputs = tuple(inputs[emb_type] for emb_type in self.emmbeding_types)
        targets = force_list(targets)
        targets = utils.split_targets_per_loss(targets, self.losses)
        losses = list()
        regularisations = list()
        additional_fetches = [(k, None) for k in additional_fetches]



        with tf.GradientTape() as tape:

            model_outputs = force_list(self.model(model_inputs))
            temp_loss_outputs = 0

            for head_idx, head_loss, head_output, head_target in zip(heads_indices, self.losses, model_outputs, targets):

                agent_input = dict(filter(lambda elem: elem[0].startswith('output_{}_'.format(head_idx)),
                                          inputs.items()))

                agent_input = list(agent_input.values())
                head_target = list(map(lambda x: tf.cast(x, tf.float32), head_target))
                loss_outputs = head_loss([head_output], agent_input, head_target)



                if LOSS_OUT_TYPE_LOSS in loss_outputs:
                    losses.extend(loss_outputs[LOSS_OUT_TYPE_LOSS])
                if LOSS_OUT_TYPE_REGULARIZATION in loss_outputs:
                    regularisations.extend(loss_outputs[LOSS_OUT_TYPE_REGULARIZATION])

                for i, fetch in enumerate(additional_fetches):
                    head_type_idx, fetch_name = fetch[0]  # fetch key is a tuple of (head_type_index, fetch_name)
                    if head_idx == head_type_idx:
                        assert fetch[1] is None  # sanity check that fetch is None
                        additional_fetches[i] = (fetch[0], loss_outputs[fetch_name])

                #########
                #temp_loss_outputs += q_loss_f(q_value_pred=head_output, target=head_target[0])
                if head_idx == 0:
                    temp_loss_outputs += v_loss_f(value_prediction=head_output, target=head_target[0])
                if head_idx == 1:
                    temp_loss_outputs += ppo_loss_f(new_policy=head_output,
                                                    actions=agent_input[0],
                                                    old_means=agent_input[1],
                                                    old_stds=agent_input[2],
                                                    advantages=head_target)





            # Total loss is losses and regularization (NOTE: order is important)
            total_loss_list = losses + regularisations
            #total_loss = tf.add_n([main_loss] + model.losses)
            total_loss = tf.add_n(total_loss_list)
            # Dan
            total_loss = temp_loss_outputs

        # Calculate gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        #gradients = tf.gradients(self.losses[0]())
        norm_unclipped_grads = tf.linalg.global_norm(gradients)

        # Gradient clipping
        if self.network_parameters.clip_gradients is not None and self.network_parameters.clip_gradients != 0:
            gradients, gradients_norm = self.clip_gradients(gradients, self.network_parameters.gradients_clipping_method,
                                                            self.network_parameters.clip_gradients)
        assert self.optimizer_type != 'LBFGS', 'LBFGS not supported'

        # Update self.accumulated_gradients depending on no_accumulation flag
        if no_accumulation:
            self.accumulated_gradients = gradients.copy()
        else:
            self.accumulated_gradients += gradients.copy()

        # Result of of additional fetches
        fetched_tensors = [fetch[1] for fetch in additional_fetches]
        # convert everything to numpy or scalar before returning
        result = (total_loss.numpy(), total_loss_list, norm_unclipped_grads.numpy(), fetched_tensors)
        return result


    def apply_gradients(self, gradients: List[np.ndarray], scaler: float=1., additional_inputs=None) -> None:
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them.
                       The gradients will be MULTIPLIED by this factor
        """
        assert self.optimizer_type != 'LBFGS'

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def apply_and_reset_gradients(self, gradients, scaler=1., additional_inputs=None):
        """
        Applies the given gradients to the network weights and resets the accumulation placeholder
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        :param additional_inputs: optional additional inputs required for when applying the gradients (e.g. batchnorm's
                                  update ops also requires the inputs)

        """
        self.apply_gradients(gradients, scaler)
        self.reset_accumulated_gradients()

    def clip_gradients(self, grads: List[np.ndarray],
                  clip_method: GradientClippingMethod,
                  clip_val: float) -> List[np.ndarray]:
        """
        Clip gradient values
        :param grads: gradients to be clipped
        :param clip_method: clipping method
        :param clip_val: clipping value. Interpreted differently depending on clipping method.
        :return: clipped gradients
        """

        if clip_method == GradientClippingMethod.ClipByGlobalNorm:
            clipped_grads, grad_norms = tf.clip_by_global_norm(grads, clip_val)

        elif clip_method == GradientClippingMethod.ClipByValue:
            clipped_grads = [tf.clip_by_value(grad, -clip_val, clip_val) for grad in grads]
        elif clip_method == GradientClippingMethod.ClipByNorm:

            clipped_grads = [tf.clip_by_norm(grad, clip_val) for grad in grads]
        else:
            raise KeyError('Unsupported gradient clipping method')
        return clipped_grads

    def _predict(self, inputs: Dict[str, np.ndarray]):
        """
        Run a forward pass of the network using the given input
        :param inputs: The input dictionary for the network. Key is name of the embedder.
        :return: The network output per each head

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """
        model_inputs = tuple(inputs[emb] for emb in self.emmbeding_types)

        assert self.middleware.__class__.__name__ != 'LSTMMiddleware'

        model_outputs = self.model(model_inputs)

        distribution_output = list(filter(lambda x: isinstance(x, Distribution), model_outputs))
        distribution_mean = [dist.mean().numpy() for dist in distribution_output]
        distribution_stddev = [dist.stddev().numpy() for dist in distribution_output]

        if distribution_output:
            value_output = list(filter(lambda x: not (isinstance(x, Distribution)), model_outputs))
            value_output = [value.numpy() for value in value_output]
            output_per_head = list(zip(value_output, distribution_mean, distribution_stddev))
            output_per_head = list(output_per_head[0])
        else:
            output_per_head = model_outputs.numpy()


        return output_per_head
        #return model_outputs


    def predict(self,
                inputs: Dict[str, np.ndarray],
                outputs: List[str]=None,
                squeeze_output: bool=True,
                initial_feed_dict: Dict[str, np.ndarray]=None) -> Tuple[np.ndarray, ...]:
        """
        Run a forward pass of the network using the given input
        :param inputs: The input dictionary for the network. Key is name of the embedder.
        :param outputs: list of outputs to return. Return all outputs if unspecified (currently not supported)
        :param squeeze_output: call squeeze_list on output if True
        :param initial_feed_dict: a dictionary of extra inputs for forward pass (currently not supported)
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """
        assert initial_feed_dict is None, "initial_feed_dict must be None"
        assert outputs is None, "outputs must be None"
        output = self._predict(inputs)
        return output

    @staticmethod
    def parallel_predict(sess: Any,
                         network_input_tuples: List[Tuple['TensorFlowArchitecture', Dict[str, np.ndarray]]]) -> \
            Tuple[np.ndarray, ...]:
        """
        :param sess: active session to use for prediction (must be None for TF2)
        :param network_input_tuples: tuple of network and corresponding input
        :return: tuple of outputs from all networks
        """
        assert sess is None
        output = [net._predict(inputs) for net, inputs in network_input_tuples]

        # output = list()
        # for net, inputs in network_input_tuples:
        #     output += net._predict(inputs)


        return output

    # def train_on_batch(self,
    #                    inputs: Dict[str, np.ndarray],
    #                    targets: List[np.ndarray],
    #                    scaler: float = 1.,
    #                    additional_fetches: list = None,
    #                    importance_weights: np.ndarray = None) -> Tuple[float, List[float], float, list]:
    #     """
    #     Given a batch of inputs (e.g. states) and targets (e.g. discounted rewards), takes a training step: i.e. runs a
    #     forward pass and backward pass of the network, accumulates the gradients and applies an optimization step to
    #     update the weights.
    #     :param inputs: environment states (observation, etc.) as well extra inputs required by loss. Shape of ndarray
    #         is (batch_size, observation_space_size) or (batch_size, observation_space_size, stack_size)
    #     :param targets: targets required by  loss (e.g. sum of discounted rewards)
    #     :param scaler: value to scale gradients by before optimizing network weights
    #     :param additional_fetches: additional fetches to calculate and return. Each fetch is specified as (int, str)
    #         tuple of head-type-index and fetch-name. The tuple is obtained from each head.
    #     :param importance_weights: ndarray of shape (batch_size,) to multiply with batch loss.
    #     :return: tuple of total_loss, losses, norm_unclipped_grads, fetched_tensors
    #         total_loss (float): sum of all head losses
    #         losses (list of float): list of all losses. The order is list of target losses followed by list
    #             of regularization losses. The specifics of losses is dependant on the network parameters
    #             (number of heads, etc.)
    #         norm_unclippsed_grads (float): global norm of all gradients before any gradient clipping is applied
    #         fetched_tensors: all values for additional_fetches
    #     """
    #     loss = self.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
    #                                      importance_weights=importance_weights)
    #     self.apply_and_reset_gradients(self.accumulated_gradients, scaler)
    #     return loss
    #
    def get_weights(self):
        """
        :return: a list of tensors containing the network weights for each layer
        """
        return self.model.get_weights()

    def set_weights(self, source_weights, new_rate=1.0):
        """
        Updates the target network weights from the given source model weights tensors
        """
        updated_target = []

        if new_rate < 0 or new_rate > 1:
            raise ValueError('new_rate parameter values should be between 0 to 1.')


        target_weights = self.model.get_weights()
        for (source_layer, target_layer) in zip(source_weights, target_weights):
            updated_target.append(new_rate * source_layer + (1 - new_rate) * target_layer)


        self.model.set_weights(updated_target)


    def set_is_training(self, state: bool) -> None:
        """
        Set the phase of the network between training and testing
        :param state: The current state (True = Training, False = Testing)
        :return: None
        """
        self.is_training = state

    def reset_internal_memory(self) -> None:
        """
        Reset any internal memory used by the network. For example, an LSTM internal state
        :return: None
        """
        assert self.middleware.__class__.__name__ != 'LSTMMiddleware', 'LSTM middleware not supported'

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collection of all checkpoints for the network (typically only one checkpoint)
        :param parent_path_suffix: path suffix of the parent of the network
            (e.g. could be name of level manager plus name of agent)
        :return: checkpoint collection for the network
        """
        name = self.name.replace('/', '.')
        savers = SaverCollection()
        if not self.distributed_training:
            savers.add(TfSaver(
                name="{}.{}".format(parent_path_suffix, name),
                model=self.model))
        return savers

