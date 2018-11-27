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
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.ndarray import NDArray

from rl_coach.architectures.architecture import Architecture
from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS, LOSS_OUT_TYPE_REGULARIZATION
from rl_coach.architectures.mxnet_components import utils
from rl_coach.architectures.mxnet_components.savers import ParameterDictSaver, OnnxSaver
from rl_coach.base_parameters import AgentParameters
from rl_coach.logger import screen
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list, squeeze_list


class MxnetArchitecture(Architecture):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, devices: List[mx.Context],
                 name: str = "", global_network=None, network_is_local: bool=True, network_is_trainable: bool=False):
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
        self._devices = self._sanitize_device_list(devices)

        self.is_chief = self.ap.task_parameters.task_index == 0
        self.network_is_global = not self.network_is_local and global_network is None
        self.distributed_training = self.network_is_global or self.network_is_local and global_network is not None

        self.optimizer_type = self.network_parameters.optimizer_type
        if self.ap.task_parameters.seed is not None:
            mx.random.seed(self.ap.task_parameters.seed)

        # Call to child class to create the model
        self.construct_model()

        self.trainer = None  # type: gluon.Trainer

    def __str__(self):
        return self.model.summary(*self._dummy_model_inputs())

    @staticmethod
    def _sanitize_device_list(devices: List[mx.Context]) -> List[mx.Context]:
        """
        Returns intersection of devices with available devices. If no intersection, returns mx.cpu()
        :param devices: list of requested devices
        :return: list of devices that are actually available
        """
        actual_device = [mx.cpu()] + [mx.gpu(i) for i in mx.test_utils.list_gpus()]
        intersection = [dev for dev in devices if dev in actual_device]
        if len(intersection) == 0:
            intersection = [mx.cpu()]
            screen.log('Requested devices {} not available. Default to CPU context.'.format(devices))
        elif len(intersection) < len(devices):
            screen.log('{} not available, using {}.'.format(
                [dev for dev in devices if dev not in intersection], intersection))
        return intersection

    def _model_grads(self, index: int=0) ->\
            Union[Generator[NDArray, NDArray, Any], Generator[List[NDArray], List[NDArray], Any]]:
        """
        Creates a copy of model gradients and returns them in a list, in the same order as collect_params()
        :param index: device index. Set to -1 to get a tuple of list of NDArrays for all devices
        :return: a generator for model gradient values
        """
        if index < 0:
            return (p.list_grad() for p in self.model.collect_params().values() if p.grad_req != 'null')
        else:
            return (p.list_grad()[index] for p in self.model.collect_params().values() if p.grad_req != 'null')

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

    def _dummy_model_inputs(self) -> Tuple[NDArray, ...]:
        """
        Creates a tuple of input arrays with correct shapes that can be used for shape inference
        of the model weights and for printing the summary
        :return: tuple of inputs for model forward pass
        """
        input_shapes = self._model_input_shapes()
        inputs = tuple(nd.zeros(tuple(shape), ctx=self._devices[0]) for shape in input_shapes)
        return inputs

    def construct_model(self) -> None:
        """
        Construct network model. Implemented by child class.
        """
        raise NotImplementedError

    def set_session(self, sess) -> None:
        """
        Initializes the model parameters and creates the model trainer.
        NOTEL Session for mxnet backend must be None.
        :param sess: must be None
        """
        assert sess is None
        # FIXME Add initializer
        self.model.collect_params().initialize(ctx=self._devices)
        # Hybridize model and losses
        self.model.hybridize()
        for l in self.losses:
            l.hybridize()

        # Pass dummy data with correct shape to trigger shape inference and full parameter initialization
        self.model(*self._dummy_model_inputs())

        if self.network_is_trainable:
            self.trainer = gluon.Trainer(
                self.model.collect_params(), optimizer=self.optimizer, update_on_kvstore=False)

    def reset_accumulated_gradients(self) -> None:
        """
        Reset model gradients as well as accumulated gradients to zero. If accumulated gradients
        have not been created yet, it constructs them on CPU.
        """
        # Set model gradients to zero
        for p in self.model.collect_params().values():
            p.zero_grad()
        # Set accumulated gradients to zero if already initialized, otherwise create a copy
        if self.accumulated_gradients:
            for a in self.accumulated_gradients:
                a *= 0
        else:
            self.accumulated_gradients = [g.copy() for g in self._model_grads()]

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
        nd_inputs = tuple(nd.array(inputs[emb], ctx=self._devices[0]) for emb in embedders)

        assert self.middleware.__class__.__name__ != 'LSTMMiddleware', "LSTM middleware not supported"

        targets = force_list(targets)
        with autograd.record():
            out_per_head = utils.split_outputs_per_head(self.model(*nd_inputs), self.model.output_heads)
            tgt_per_loss = utils.split_targets_per_loss(targets, self.losses)

            losses = list()
            regularizations = list()
            additional_fetches = [(k, None) for k in additional_fetches]
            for h, h_loss, h_out, l_tgt in zip(self.model.output_heads, self.losses, out_per_head, tgt_per_loss):
                l_in = utils.get_loss_agent_inputs(inputs, head_type_idx=h.head_type_idx, loss=h_loss)
                # Align arguments with loss.loss_forward and convert to NDArray
                l_args = utils.to_mx_ndarray(utils.align_loss_args(h_out, l_in, l_tgt, h_loss), h_out[0].context)
                # Calculate loss and all auxiliary outputs
                loss_outputs = utils.loss_output_dict(utils.to_list(h_loss(*l_args)), h_loss.output_schema)
                if LOSS_OUT_TYPE_LOSS in loss_outputs:
                    losses.extend(loss_outputs[LOSS_OUT_TYPE_LOSS])
                if LOSS_OUT_TYPE_REGULARIZATION in loss_outputs:
                    regularizations.extend(loss_outputs[LOSS_OUT_TYPE_REGULARIZATION])
                # Set additional fetches
                for i, fetch in enumerate(additional_fetches):
                    head_type_idx, fetch_name = fetch[0]  # fetch key is a tuple of (head_type_index, fetch_name)
                    if head_type_idx == h.head_type_idx:
                        assert fetch[1] is None  # sanity check that fetch is None
                        additional_fetches[i] = (fetch[0], loss_outputs[fetch_name])

            # Total loss is losses and regularization (NOTE: order is important)
            total_loss_list = losses + regularizations
            total_loss = nd.add_n(*total_loss_list)

        # Calculate gradients
        total_loss.backward()

        assert self.optimizer_type != 'LBFGS', 'LBFGS not supported'

        # allreduce gradients from all contexts
        self.trainer.allreduce_grads()

        model_grads_cpy = [g.copy() for g in self._model_grads()]
        # Calculate global norm of gradients
        # FIXME global norm is returned even when not used for clipping! Is this necessary?
        # FIXME global norm might be calculated twice if clipping method is global norm
        norm_unclipped_grads = utils.global_norm(model_grads_cpy)

        # Clip gradients
        if self.network_parameters.clip_gradients:
            utils.clip_grad(
                model_grads_cpy,
                clip_method=self.network_parameters.gradients_clipping_method,
                clip_val=self.network_parameters.clip_gradients,
                inplace=True)

        # Update self.accumulated_gradients depending on no_accumulation flag
        if no_accumulation:
            for acc_grad, model_grad in zip(self.accumulated_gradients, model_grads_cpy):
                acc_grad[:] = model_grad
        else:
            for acc_grad, model_grad in zip(self.accumulated_gradients, model_grads_cpy):
                acc_grad += model_grad

        # result of of additional fetches
        fetched_tensors = [fetch[1] for fetch in additional_fetches]

        # convert everything to numpy or scalar before returning
        result = utils.asnumpy_or_asscalar((total_loss, total_loss_list, norm_unclipped_grads, fetched_tensors))
        return result

    def apply_and_reset_gradients(self, gradients: List[np.ndarray], scaler: float=1.) -> None:
        """
        Applies the given gradients to the network weights and resets accumulated gradients to zero
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        self.apply_gradients(gradients, scaler)
        self.reset_accumulated_gradients()

    def apply_gradients(self, gradients: List[np.ndarray], scaler: float=1.) -> None:
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them.
                       The gradients will be MULTIPLIED by this factor
        """
        assert self.optimizer_type != 'LBFGS'

        batch_size = 1
        if self.distributed_training and not self.network_parameters.async_training:
            # rescale the gradients so that they average out with the gradients from the other workers
            if self.network_parameters.scale_down_gradients_by_number_of_workers_for_sync_training:
                batch_size = self.ap.task_parameters.num_training_tasks

        # set parameter gradients to gradients passed in
        for param_grad, gradient in zip(self._model_grads(-1), gradients):
            for pg in param_grad:
                pg[:] = gradient
        # update gradients
        self.trainer.update(batch_size=batch_size)

    def _predict(self, inputs: Dict[str, np.ndarray]) -> Tuple[NDArray, ...]:
        """
        Run a forward pass of the network using the given input
        :param inputs: The input dictionary for the network. Key is name of the embedder.
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """
        embedders = [emb.embedder_name for emb in self.model.nets[0].input_embedders]
        nd_inputs = tuple(nd.array(inputs[emb], ctx=self._devices[0]) for emb in embedders)

        assert self.middleware.__class__.__name__ != 'LSTMMiddleware'

        output = self.model(*nd_inputs)
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
        output = list(o.asnumpy() for o in output)
        if squeeze_output:
            output = squeeze_list(output)
        return output

    @staticmethod
    def parallel_predict(sess: Any,
                         network_input_tuples: List[Tuple['MxnetArchitecture', Dict[str, np.ndarray]]]) -> \
            Tuple[np.ndarray, ...]:
        """
        :param sess: active session to use for prediction (must be None for MXNet)
        :param network_input_tuples: tuple of network and corresponding input
        :return: tuple of outputs from all networks
        """
        assert sess is None
        output = list()
        for net, inputs in network_input_tuples:
            output += net._predict(inputs)
        return tuple(o.asnumpy() for o in output)

    def train_on_batch(self,
                       inputs: Dict[str, np.ndarray],
                       targets: List[np.ndarray],
                       scaler: float = 1.,
                       additional_fetches: list = None,
                       importance_weights: np.ndarray = None) -> Tuple[float, List[float], float, list]:
        """
        Given a batch of inputs (e.g. states) and targets (e.g. discounted rewards), takes a training step: i.e. runs a
        forward pass and backward pass of the network, accumulates the gradients and applies an optimization step to
        update the weights.
        :param inputs: environment states (observation, etc.) as well extra inputs required by loss. Shape of ndarray
            is (batch_size, observation_space_size) or (batch_size, observation_space_size, stack_size)
        :param targets: targets required by  loss (e.g. sum of discounted rewards)
        :param scaler: value to scale gradients by before optimizing network weights
        :param additional_fetches: additional fetches to calculate and return. Each fetch is specified as (int, str)
            tuple of head-type-index and fetch-name. The tuple is obtained from each head.
        :param importance_weights: ndarray of shape (batch_size,) to multiply with batch loss.
        :return: tuple of total_loss, losses, norm_unclipped_grads, fetched_tensors
            total_loss (float): sum of all head losses
            losses (list of float): list of all losses. The order is list of target losses followed by list
                of regularization losses. The specifics of losses is dependant on the network parameters
                (number of heads, etc.)
            norm_unclippsed_grads (float): global norm of all gradients before any gradient clipping is applied
            fetched_tensors: all values for additional_fetches
        """
        loss = self.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
                                         importance_weights=importance_weights)
        self.apply_and_reset_gradients(self.accumulated_gradients, scaler)
        return loss

    def get_weights(self) -> gluon.ParameterDict:
        """
        :return: a ParameterDict containing all network weights
        """
        return self.model.collect_params()

    def set_weights(self, weights: gluon.ParameterDict, new_rate: float=1.0) -> None:
        """
        Sets the network weights from the given ParameterDict
        :param new_rate: ratio for adding new and old weight values: val=rate*weights + (1-rate)*old_weights
        """
        old_weights = self.model.collect_params()
        for name, p in weights.items():
            name = name[len(weights.prefix):]  # Strip prefix
            old_p = old_weights[old_weights.prefix + name]  # Add prefix
            old_p.set_data(new_rate * p._reduce() + (1 - new_rate) * old_p._reduce())

    def get_variable_value(self, variable: Union[gluon.Parameter, NDArray]) -> np.ndarray:
        """
        Get the value of a variable
        :param variable: the variable
        :return: the value of the variable
        """
        if isinstance(variable, gluon.Parameter):
            variable = variable._reduce().asnumpy()
        if isinstance(variable, NDArray):
            return variable.asnumpy()
        return variable

    def set_variable_value(self, assign_op: callable, value: Any, placeholder=None) -> None:
        """
        Updates value of a variable.
        :param assign_op: a callable assign function for setting the variable
        :param value: a value to set the variable to
        :param placeholder: unused (placeholder in symbolic framework backends)
        """
        assert callable(assign_op)
        assign_op(value)

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
        savers = SaverCollection(ParameterDictSaver(
            name="{}.{}".format(parent_path_suffix, name),
            param_dict=self.model.collect_params()))
        if self.ap.task_parameters.export_onnx_graph:
            savers.add(OnnxSaver(
                name="{}.{}.onnx".format(parent_path_suffix, name),
                model=self.model,
                input_shapes=self._model_input_shapes()))
        return savers
