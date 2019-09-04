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

from rl_coach.architectures.architecture import Architecture

from rl_coach.base_parameters import AgentParameters
from rl_coach.logger import screen
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list, squeeze_list
from rl_coach.architectures.tensorflow_components import utils
from rl_coach.architectures.tensorflow_components.heads import Head
from rl_coach.architectures.tensorflow_components.losses import HeadLoss
from rl_coach.core_types import GradientClippingMethod
from rl_coach.architectures.tensorflow_components.savers import TfSaver



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
        self.losses = []  # type: List[HeadLoss]
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
        if self.ap.task_parameters.seed is not None:
            tf.compat.v1.set_random_seed(self.ap.task_parameters.seed)

        # Call to child class to create the model
        self.construct_model()
        self.trainer = None

    def __str__(self):
        return self.model.summary(self._dummy_model_inputs())


    def _model_input_shapes(self) -> List[List[int]]:
        """
        Create a list of input array shapes
        :return: type of input shapes
        """
        allowed_inputs = copy.copy(self.spaces.state.sub_spaces)
        allowed_inputs["action"] = copy.copy(self.spaces.action)
        allowed_inputs["goal"] = copy.copy(self.spaces.goal)
        embedders = self.model.nets[0].input_embedders
        return list([1] + allowed_inputs[emb.embedder_name].shape.tolist() for emb in embedders)

    def _dummy_model_inputs(self):
        """
        Creates a tuple of input arrays with correct shapes that can be used for shape inference
        of the model weights and for printing the summary
        :return: tuple of inputs for model forward pass
        """
        input_shapes = self._model_input_shapes()
        inputs = tuple(np.zeros(tuple(shape)) for shape in input_shapes) #make this the same type as the actual input
        return inputs


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
        # Pass dummy data with correct shape to trigger shape inference and full parameter initialization
        self.model(self._dummy_model_inputs())
        self.model.summary()


    def reset_accumulated_gradients(self) -> None:
        """
        Reset the gradients accumulation
        """

        if self.accumulated_gradients is None:
            self.accumulated_gradients = self.model.get_weights().copy()

        self.accumulated_gradients = list(map(lambda grad: grad * 0, self.accumulated_gradients))

        # for ix, grad in enumerate(self.accumulated_gradients):
        #     self.accumulated_gradients[ix] = grad * 0





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
        if self.accumulated_gradients is None:
            self.reset_accumulated_gradients()

        embedders = [emb.embedder_name for emb in self.model.nets[0].input_embedders]
        #_inputs = np.squeeze(tuple(inputs[emb] for emb in embedders))
        _inputs = tuple(inputs[emb] for emb in embedders)
        assert self.middleware.__class__.__name__ != 'LSTMMiddleware', "LSTM middleware not supported"
        targets = force_list(targets)

        with tf.GradientTape() as tape:
            out_per_head = self.model(_inputs)
            losses = list()
            regularizations = list()
            additional_fetches = [(k, None) for k in additional_fetches]
            for h, h_loss, h_out, l_tgt in zip(self.model.output_heads, self.losses, out_per_head, targets):
                #l_in = utils.get_loss_agent_inputs(inputs, head_type_idx=h.head_type_idx, loss=h_loss)
                loss_outputs = h_loss(h_out, l_tgt)
                losses.append(loss_outputs)
                for i, fetch in enumerate(additional_fetches):
                    head_type_idx, fetch_name = fetch[0]  # fetch key is a tuple of (head_type_index, fetch_name)
                    if head_type_idx == h.head_type_idx:
                        assert fetch[1] is None  # sanity check that fetch is None
                        additional_fetches[i] = (fetch[0], loss_outputs[fetch_name])

            # Total loss is losses and regularization (NOTE: order is important)
            total_loss_list = losses + regularizations
            #total_loss = tf.add_n([main_loss] + model.losses)
            total_loss = tf.add_n(total_loss_list)

        # Calculate gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        norm_unclipped_grads = tf.linalg.global_norm(gradients)
        # gradient clipping
        if self.network_parameters.clip_gradients is not None and self.network_parameters.clip_gradients != 0:
            gradients, gradients_norm = self.clip_gradients(gradients, self.network_parameters.gradients_clipping_method,
                                                            self.network_parameters.clip_gradients)
        assert self.optimizer_type != 'LBFGS', 'LBFGS not supported'

        # Update self.accumulated_gradients depending on no_accumulation flag
        if no_accumulation:
            for acc_grad, model_grad in zip(self.accumulated_gradients, gradients):
                acc_grad[:] = model_grad
        else:
            for acc_grad, model_grad in zip(self.accumulated_gradients, gradients):
                acc_grad += model_grad

        # result of of additional fetches
        fetched_tensors = [fetch[1] for fetch in additional_fetches]
        # convert everything to numpy or scalar before returning
        result = (total_loss.numpy(), total_loss_list, norm_unclipped_grads.numpy(), fetched_tensors)
        return result


    def apply_gradients(self, gradients: List[np.ndarray], scaler: float=1.) -> None:
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them.
                       The gradients will be MULTIPLIED by this factor
        """
        assert self.optimizer_type != 'LBFGS'

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



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
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """
        embedders = [emb.embedder_name for emb in self.model.nets[0].input_embedders]
        #_inputs = np.squeeze(tuple(inputs[emb] for emb in embedders))
        _inputs = tuple(inputs[emb] for emb in embedders)

        #tf.data.Dataset.from_tensor_slices(_inputs)

        # # can also
        # v = np.array([[1., 2., 3., 4.], [4., 4., 5., 6.], [2., 3., 3., 3.]])
        # self.model(v)
        assert self.middleware.__class__.__name__ != 'LSTMMiddleware'

        output = self.model(_inputs)
        return output

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
        output = list(o[0].numpy() for o in output)
        if squeeze_output:
            output = squeeze_list(output)
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
        output = list()
        for net, inputs in network_input_tuples:
            output += net._predict(inputs)
            #output.append(net._predict(inputs))

        return tuple(o[0].numpy() for o in output)

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

