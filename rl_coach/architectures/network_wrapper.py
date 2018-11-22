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

from typing import List, Tuple

from rl_coach.base_parameters import Frameworks, AgentParameters
from rl_coach.logger import failed_imports
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list
try:
    import tensorflow as tf
    from rl_coach.architectures.tensorflow_components.general_network import GeneralTensorFlowNetwork
except ImportError:
    failed_imports.append("tensorflow")

try:
    import mxnet as mx
    from rl_coach.architectures.mxnet_components.general_network import GeneralMxnetNetwork
except ImportError:
    failed_imports.append("mxnet")


class NetworkWrapper(object):
    """
    The network wrapper contains multiple copies of the same network, each one with a different set of weights which is
    updating in a different time scale. The network wrapper will always contain an online network.
    It will contain an additional slow updating target network if it was requested by the user,
    and it will contain a global network shared between different workers, if Coach is run in a single-node
    multi-process distributed mode. The network wrapper contains functionality for managing these networks and syncing
    between them.
    """
    def __init__(self, agent_parameters: AgentParameters, has_target: bool, has_global: bool, name: str,
                 spaces: SpacesDefinition, replicated_device=None, worker_device=None):
        self.ap = agent_parameters
        self.network_parameters = self.ap.network_wrappers[name]
        self.has_target = has_target
        self.has_global = has_global
        self.name = name
        self.sess = None

        if self.network_parameters.framework == Frameworks.tensorflow:
            if "tensorflow" not in failed_imports:
                general_network = GeneralTensorFlowNetwork.construct
            else:
                raise Exception('Install tensorflow before using it as framework')
        elif self.network_parameters.framework == Frameworks.mxnet:
            if "mxnet" not in failed_imports:
                general_network = GeneralMxnetNetwork.construct
            else:
                raise Exception('Install mxnet before using it as framework')
        else:
            raise Exception("{} Framework is not supported"
                            .format(Frameworks().to_string(self.network_parameters.framework)))

        variable_scope = "{}/{}".format(self.ap.full_name_id, name)

        # Global network - the main network shared between threads
        self.global_network = None
        if self.has_global:
            # we assign the parameters of this network on the parameters server
            self.global_network = general_network(variable_scope=variable_scope,
                                                  devices=force_list(replicated_device),
                                                  agent_parameters=agent_parameters,
                                                  name='{}/global'.format(name),
                                                  global_network=None,
                                                  network_is_local=False,
                                                  spaces=spaces,
                                                  network_is_trainable=True)

        # Online network - local copy of the main network used for playing
        self.online_network = None
        self.online_network = general_network(variable_scope=variable_scope,
                                              devices=force_list(worker_device),
                                              agent_parameters=agent_parameters,
                                              name='{}/online'.format(name),
                                              global_network=self.global_network,
                                              network_is_local=True,
                                              spaces=spaces,
                                              network_is_trainable=True)

        # Target network - a local, slow updating network used for stabilizing the learning
        self.target_network = None
        if self.has_target:
            self.target_network = general_network(variable_scope=variable_scope,
                                                  devices=force_list(worker_device),
                                                  agent_parameters=agent_parameters,
                                                  name='{}/target'.format(name),
                                                  global_network=self.global_network,
                                                  network_is_local=True,
                                                  spaces=spaces,
                                                  network_is_trainable=False)

    def sync(self):
        """
        Initializes the weights of the networks to match each other

        :return:
        """
        self.update_online_network()
        self.update_target_network()

    def update_target_network(self, rate=1.0):
        """
        Copy weights: online network >>> target network

        :param rate: the rate of copying the weights - 1 for copying exactly
        """
        if self.target_network:
            self.target_network.set_weights(self.online_network.get_weights(), rate)

    def update_online_network(self, rate=1.0):
        """
        Copy weights: global network >>> online network

        :param rate: the rate of copying the weights - 1 for copying exactly
        """
        if self.global_network:
            self.online_network.set_weights(self.global_network.get_weights(), rate)

    def apply_gradients_to_global_network(self, gradients=None):
        """
        Apply gradients from the online network on the global network

        :param gradients: optional gradients that will be used instead of teh accumulated gradients
        :return:
        """
        if gradients is None:
            gradients = self.online_network.accumulated_gradients
        if self.network_parameters.shared_optimizer:
            self.global_network.apply_gradients(gradients)
        else:
            self.online_network.apply_gradients(gradients)

    def apply_gradients_to_online_network(self, gradients=None):
        """
        Apply gradients from the online network on itself

        :return:
        """
        if gradients is None:
            gradients = self.online_network.accumulated_gradients
        self.online_network.apply_gradients(gradients)

    def train_and_sync_networks(self, inputs, targets, additional_fetches=[], importance_weights=None):
        """
        A generic training function that enables multi-threading training using a global network if necessary.

        :param inputs: The inputs for the network.
        :param targets: The targets corresponding to the given inputs
        :param additional_fetches: Any additional tensor the user wants to fetch
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :return: The loss of the training iteration
        """
        result = self.online_network.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
                                                          importance_weights=importance_weights, no_accumulation=True)
        self.apply_gradients_and_sync_networks(reset_gradients=False)
        return result

    def apply_gradients_and_sync_networks(self, reset_gradients=True):
        """
        Applies the gradients accumulated in the online network to the global network or to itself and syncs the
        networks if necessary

        :param reset_gradients: If set to True, the accumulated gradients wont be reset to 0 after applying them to
                                the network. this is useful when the accumulated gradients are overwritten instead
                                if accumulated by the accumulate_gradients function. this allows reducing time
                                complexity for this function by around 10%
        """
        if self.global_network:
            self.apply_gradients_to_global_network()
            if reset_gradients:
                self.online_network.reset_accumulated_gradients()
            self.update_online_network()
        else:
            if reset_gradients:
                self.online_network.apply_and_reset_gradients(self.online_network.accumulated_gradients)
            else:
                self.online_network.apply_gradients(self.online_network.accumulated_gradients)

    def parallel_prediction(self, network_input_tuples: List[Tuple]):
        """
        Run several network prediction in parallel. Currently this only supports running each of the network once.

        :param network_input_tuples: a list of tuples where the first element is the network (online_network,
                                     target_network or global_network) and the second element is the inputs
        :return: the outputs of all the networks in the same order as the inputs were given
        """
        return type(self.online_network).parallel_predict(self.sess, network_input_tuples)

    def set_is_training(self, state: bool):
        """
        Set the phase of the network between training and testing

        :param state: The current state (True = Training, False = Testing)
        :return: None
        """
        self.online_network.set_is_training(state)
        if self.has_target:
            self.target_network.set_is_training(state)

    def set_session(self, sess):
        self.sess = sess
        self.online_network.set_session(sess)
        if self.global_network:
            self.global_network.set_session(sess)
        if self.target_network:
            self.target_network.set_session(sess)

    def __str__(self):
        sub_networks = []
        if self.global_network:
            sub_networks.append("global network")
        if self.online_network:
            sub_networks.append("online network")
        if self.target_network:
            sub_networks.append("target network")

        result = []
        result.append("Network: {}, Copies: {} ({})".format(self.name, len(sub_networks), ' | '.join(sub_networks)))
        result.append("-"*len(result[-1]))
        result.append(str(self.online_network))
        result.append("")
        return '\n'.join(result)

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collect all of network's savers for global or online network
        Note: global, online, and target network are all copies fo the same network which parameters that are
            updated at different rates. So we only need to save one of the networks; the one that holds the most
            recent parameters. target network is created for some agents and used for stabilizing training by
            updating parameters from online network at a slower rate. As a result, target network never contains
            the most recent set of parameters. In single-worker training, no global network is created and online
            network contains the most recent parameters. In vertical distributed training with more than one worker,
            global network is updated by all workers and contains the most recent parameters.
            Therefore preference is given to global network if it exists, otherwise online network is used
            for saving.
        :param parent_path_suffix: path suffix of the parent of the network wrapper
            (e.g. could be name of level manager plus name of agent)
        :return: collection of all checkpoint objects
        """
        if self.global_network:
            savers = self.global_network.collect_savers(parent_path_suffix)
        else:
            savers = self.online_network.collect_savers(parent_path_suffix)
        return savers
