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

from typing import Any, Dict, List, Tuple

import numpy as np

from rl_coach.base_parameters import AgentParameters
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition


class Architecture(object):
    @staticmethod
    def construct(variable_scope: str, devices: List[str], *args, **kwargs) -> 'Architecture':
        """
        Construct a network class using the provided variable scope and on requested devices
        :param variable_scope: string specifying variable scope under which to create network variables
        :param devices: list of devices (can be list of Device objects, or string for TF distributed)
        :param args: all other arguments for class initializer
        :param kwargs: all other keyword arguments for class initializer
        :return: an object which is a child of Architecture
        """
        raise NotImplementedError

    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str= ""):
        """
        Creates a neural network 'architecture', that can be trained and used for inference.

        :param agent_parameters: the agent parameters
        :param spaces: the spaces (observation, action, etc.) definition of the agent
        :param name: the name of the network
        """
        self.spaces = spaces
        self.name = name
        self.network_wrapper_name = self.name.split('/')[0]  # e.g. 'main/online' --> 'main'
        self.full_name = "{}/{}".format(agent_parameters.full_name_id, name)
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]
        self.batch_size = self.network_parameters.batch_size
        self.learning_rate = self.network_parameters.learning_rate
        self.optimizer = None
        self.ap = agent_parameters

    def predict(self,
                inputs: Dict[str, np.ndarray],
                outputs: List[Any] = None,
                squeeze_output: bool = True,
                initial_feed_dict: Dict[Any, np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Given input observations, use the model to make predictions (e.g. action or value).

        :param inputs: current state (i.e. observations, measurements, goals, etc.)
            (e.g. `{'observation': numpy.ndarray}` of shape (batch_size, observation_space_size))
        :param outputs: list of outputs to return. Return all outputs if unspecified. Type of the list elements
            depends on the framework backend.
        :param squeeze_output: call squeeze_list on output before returning if True
        :param initial_feed_dict: a dictionary of extra inputs for forward pass.
        :return: predictions of action or value of shape (batch_size, action_space_size) for action predictions)
        """
        raise NotImplementedError

    @staticmethod
    def parallel_predict(sess: Any,
                         network_input_tuples: List[Tuple['Architecture', Dict[str, np.ndarray]]]) -> \
            Tuple[np.ndarray, ...]:
        """
        :param sess: active session to use for prediction
        :param network_input_tuples: tuple of network and corresponding input
        :return: list or tuple of outputs from all networks
        """
        raise NotImplementedError

    def train_on_batch(self,
                       inputs: Dict[str, np.ndarray],
                       targets: List[np.ndarray],
                       scaler: float=1.,
                       additional_fetches: list=None,
                       importance_weights: np.ndarray=None) -> Tuple[float, List[float], float, list]:
        """
        Given a batch of inputs (e.g. states) and targets (e.g. discounted rewards), takes a training step: i.e. runs a
        forward pass and backward pass of the network, accumulates the gradients and applies an optimization step to
        update the weights.
        Calls `accumulate_gradients` followed by `apply_and_reset_gradients`.
        Note: Currently an unused method.

        :param inputs: typically the environment states (but can also contain other data necessary for loss).
            (e.g. `{'observation': numpy.ndarray}` with `observation` of shape (batch_size, observation_space_size) or
            (batch_size, observation_space_size, stack_size) or
            `{'observation': numpy.ndarray, 'output_0_0': numpy.ndarray}` with `output_0_0` of shape (batch_size,))
        :param targets: target values of shape (batch_size, ). For example discounted rewards for value network
            for calculating the value-network loss would be a target. Length of list and order of arrays in
            the list matches that of network losses which are defined by network parameters
        :param scaler: value to scale gradients by before optimizing network weights
        :param additional_fetches: list of additional values to fetch and return. The type of each list
            element is framework dependent.
        :param importance_weights: ndarray of shape (batch_size,) to multiply with batch loss.
        :return: tuple of total_loss, losses, norm_unclipped_grads, fetched_tensors
            total_loss (float): sum of all head losses
            losses (list of float): list of all losses. The order is list of target losses followed by list
                of regularization losses. The specifics of losses is dependant on the network parameters
                (number of heads, etc.)
            norm_unclippsed_grads (float): global norm of all gradients before any gradient clipping is applied
            fetched_tensors: all values for additional_fetches
        """
        raise NotImplementedError

    def get_weights(self) -> List[np.ndarray]:
        """
        Gets model weights as a list of ndarrays. It is used for synchronizing weight between two identical networks.

        :return: list weights as ndarray
        """
        raise NotImplementedError

    def set_weights(self, weights: List[np.ndarray], rate: float=1.0) -> None:
        """
        Sets model weights for provided layer parameters.

        :param weights: list of model weights in the same order as received in get_weights
        :param rate: controls the mixture of given weight values versus old weight values.
            i.e. new_weight = rate * given_weight + (1 - rate) * old_weight
        :return: None
        """
        raise NotImplementedError

    def reset_accumulated_gradients(self) -> None:
        """
        Sets gradient of all parameters to 0.

        Once gradients are reset, they must be accessible by `accumulated_gradients` property of this class,
        which must return a list of numpy ndarrays. Child class must ensure that `accumulated_gradients` is set.
        """
        raise NotImplementedError

    def accumulate_gradients(self,
                             inputs: Dict[str, np.ndarray],
                             targets: List[np.ndarray],
                             additional_fetches: list=None,
                             importance_weights: np.ndarray=None,
                             no_accumulation: bool=False) -> Tuple[float, List[float], float, list]:
        """
        Given a batch of inputs (i.e. states) and targets (e.g. discounted rewards), computes and accumulates the
        gradients for model parameters. Will run forward and backward pass to compute gradients, clip the gradient
        values if required and then accumulate gradients from all learners. It does not update the model weights,
        that's performed in `apply_and_reset_gradients` method.

        Once gradients are accumulated, they are accessed by `accumulated_gradients` property of this class.å

        :param inputs: typically the environment states (but can also contain other data for loss)
            (e.g. `{'observation': numpy.ndarray}` with `observation` of shape (batch_size, observation_space_size) or
             (batch_size, observation_space_size, stack_size) or
            `{'observation': numpy.ndarray, 'output_0_0': numpy.ndarray}` with `output_0_0` of shape (batch_size,))
        :param targets: targets for calculating loss. For example discounted rewards for value network
            for calculating the value-network loss would be a target. Length of list and order of arrays in
            the list matches that of network losses which are defined by network parameters
        :param additional_fetches: list of additional values to fetch and return. The type of each list
            element is framework dependent.
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
        raise NotImplementedError

    def apply_and_reset_gradients(self, gradients: List[np.ndarray], scaler: float=1.) -> None:
        """
        Applies the given gradients to the network weights and resets the gradient accumulations.
        Has the same impact as calling `apply_gradients`, then `reset_accumulated_gradients`.

        :param gradients: gradients for the parameter weights, taken from `accumulated_gradients` property
            of an identical network (either self or another identical network)
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        raise NotImplementedError

    def apply_gradients(self, gradients: List[np.ndarray], scaler: float=1.) -> None:
        """
        Applies the given gradients to the network weights.
        Will be performed sync or async depending on `network_parameters.async_training`

        :param gradients: gradients for the parameter weights, taken from `accumulated_gradients` property
            of an identical network (either self or another identical network)
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        raise NotImplementedError

    def get_variable_value(self, variable: Any) -> np.ndarray:
        """
        Gets value of a specified variable. Type of variable is dependant on the framework.
        Example of a variable is head.kl_coefficient, which could be a symbol for evaluation
        or could be a string representing the value.

        :param variable: variable of interest
        :return: value of the specified variable
        """
        raise NotImplementedError

    def set_variable_value(self, assign_op: Any, value: np.ndarray, placeholder: Any):
        """
        Updates the value of a specified variable. Type of assign_op is dependant on the framework
        and is a unique identifier for assigning value to a variable. For example an agent may use
        head.assign_kl_coefficient. There is a one to one mapping between assign_op and placeholder
        (in the example above, placeholder would be head.kl_coefficient_ph).

        :param assign_op: a parameter representing the operation for assigning value to a specific variable
        :param value: value of the specified variable used for update
        :param placeholder: a placeholder for binding the value to assign_op.
        """
        raise NotImplementedError

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collection of all savers for the network (typically only one saver for network and one for ONNX export)
        :param parent_path_suffix: path suffix of the parent of the network
            (e.g. could be name of level manager plus name of agent)
        :return: saver collection for the network
        """
        raise NotImplementedError
