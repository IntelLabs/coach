#
# Copyright (c) 2019 Intel Corporation
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import List
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError
from rl_coach.base_parameters import AgentParameters

from rl_coach.spaces import SpacesDefinition
from rl_coach.architectures.tensorflow_components.architecture import TensorFlowArchitecture
from rl_coach.architectures.tensorflow_components.dnn_model import create_full_model
from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss
from rl_coach.architectures.tensorflow_components.losses.q_loss import QLoss
from rl_coach.architectures.tensorflow_components.losses.v_loss import VLoss
from rl_coach.architectures.tensorflow_components.losses.ppo_loss import PPOLoss
from rl_coach.architectures.head_parameters import PPOHeadParameters, VHeadParameters, QHeadParameters


class GeneralTensorFlowNetwork(TensorFlowArchitecture):
    """
    A generalized version of all possible networks implemented using tensorflow along with the optimizer and loss.
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
        # TODO: TF2 place holder for distributed training in TensorFlow
        generalized_network = GeneralTensorFlowNetwork(*args, **kwargs)
        generalized_network.model.summary()

        #generalized_network.model.compile(loss=[v_loss_f, ppo_loss_f], loss_weights=[1, 1])
        keras.utils.plot_model(generalized_network.model,
                               expand_nested=True,
                               show_shapes=True,
                               to_file='model_plot_new.png')
        img = mpimg.imread('model_plot_new.png')
        plt.imshow(img)
        plt.show()

        return generalized_network

    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
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

        super().__init__(agent_parameters, spaces, name, global_network, network_is_local, network_is_trainable)

        self.global_network = global_network

        self.network_wrapper_name = name.split('/')[0]

        network_name = name.split('/')[0] + '_' + name.split('/')[1]

        network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]

        if len(network_parameters.input_embedders_parameters) == 0:
            raise ValueError("At least one input type should be defined")

        if len(network_parameters.heads_parameters) == 0:
            raise ValueError("At least one output type should be defined")

        if network_parameters.middleware_parameters is None:
            raise ValueError("Exactly one middleware type should be defined")

        if network_parameters.use_separate_networks_per_head:
            num_heads_per_network = 1
            num_networks = len(network_parameters.heads_parameters)
        else:
            num_heads_per_network = len(network_parameters.heads_parameters)
            num_networks = 1

        self.model = create_full_model(num_networks=num_networks,
                                       num_heads_per_network=num_heads_per_network,
                                       network_is_local=network_is_local,
                                       network_name=network_name,
                                       agent_parameters=agent_parameters,
                                       network_parameters=network_parameters,
                                       spaces=spaces)

        self.losses = list()
        for index, loss_params in enumerate(network_parameters.heads_parameters):
            loss = self._get_loss(agent_parameters=agent_parameters,
                                  loss_params=loss_params,
                                  network_name=loss_params.name,
                                  num_actions=spaces.action.shape[0],
                                  head_idx=index,
                                  loss_weight=loss_params.loss_weight)
            self.losses.append(loss)

        self.optimizer = self._get_optimizer(network_parameters)
        self.network_parameters = agent_parameters.network_wrappers[self.network_wrapper_name]

    def _get_optimizer(self, network_parameters):
        """
        :param network_parameters: class containing the relevant optimizer parameters.
        :return:  an instantiation of tensorFlow optimizer with the configured parameters
        """
        if network_parameters.optimizer_type == 'Adam':
            optimizer = keras.optimizers.Adam(
                lr=network_parameters.learning_rate,
                beta_1=network_parameters.adam_optimizer_beta1,
                beta_2=network_parameters.adam_optimizer_beta2,
                epsilon=network_parameters.optimizer_epsilon)

        elif network_parameters.optimizer_type == 'RMSProp':
            optimizer = keras.optimizers.RMSprop(
                lr=network_parameters.learning_rate,
                decay=network_parameters.rms_prop_optimizer_decay,
                epsilon=network_parameters.optimizer_epsilon)

        elif network_parameters.optimizer_type == 'LBFGS':
            raise NotImplementedError(' Could not find updated LBFGS implementation')  # TODO: TF2 to update function
        else:
            raise Exception("{} is not a valid optimizer type".format(self.network_parameters.optimizer_type))

        return optimizer

    def _get_loss(self,
                  agent_parameters,
                  loss_params,
                  network_name: str,
                  num_actions,
                  head_idx,
                  loss_weight) -> HeadLoss:
        """
        Given a loss type, creates the loss and returns it
        :param loss_params: the parameters of the loss to create
        :param head_idx: the head index
        :param network_name: name of the network
        :return: loss block
        """
        if isinstance(loss_params, QHeadParameters):
            loss = QLoss(network_name=network_name,
                         head_idx=head_idx,
                         loss_type=MeanSquaredError,
                         loss_weight=loss_weight)

        elif isinstance(loss_params, VHeadParameters):
            loss = VLoss(network_name=network_name,
                         head_idx=head_idx,
                         loss_type=MeanSquaredError,
                         loss_weight=loss_weight)

        elif isinstance(loss_params, PPOHeadParameters):
            loss = PPOLoss(network_name=network_name,
                           agent_parameters=agent_parameters,
                           num_actions=num_actions,
                           head_idx=head_idx,
                           loss_type=MeanSquaredError,
                           loss_weight=loss_weight)

        else:
            raise KeyError('Unsupported loss type: {}'.format(type(loss_params)))

        return loss

    @property
    def output_heads(self):
        output_heads = list(map(lambda sub_model: sub_model.layers[-1], self.model.layers[1:]))
        return output_heads


