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
from rl_coach.spaces import SpacesDefinition

try:
    import tensorflow as tf
    from rl_coach.architectures.tensorflow_components.general_network import GeneralTensorFlowNetwork
except ImportError:
    failed_imports.append("TensorFlow")


class NetworkWrapper(object):
    """
    Contains multiple networks and managers syncing and gradient updates
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
            general_network = GeneralTensorFlowNetwork
        else:
            raise Exception("{} Framework is not supported"
                            .format(Frameworks().to_string(self.network_parameters.framework)))

        with tf.variable_scope("{}/{}".format(self.ap.full_name_id, name)):

            # Global network - the main network shared between threads
            self.global_network = None
            if self.has_global:
                # we assign the parameters of this network on the parameters server
                with tf.device(replicated_device):
                    self.global_network = general_network(agent_parameters=agent_parameters,
                                                          name='{}/global'.format(name),
                                                          global_network=None,
                                                          network_is_local=False,
                                                          spaces=spaces,
                                                          network_is_trainable=True)

            # Online network - local copy of the main network used for playing
            self.online_network = None
            with tf.device(worker_device):
                self.online_network = general_network(agent_parameters=agent_parameters,
                                                      name='{}/online'.format(name),
                                                      global_network=self.global_network,
                                                      network_is_local=True,
                                                      spaces=spaces,
                                                      network_is_trainable=True)

            # Target network - a local, slow updating network used for stabilizing the learning
            self.target_network = None
            if self.has_target:
                with tf.device(worker_device):
                    self.target_network = general_network(agent_parameters=agent_parameters,
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
        feed_dict = {}
        fetches = []

        for idx, (network, input) in enumerate(network_input_tuples):
            feed_dict.update(network.create_feed_dict(input))
            fetches += network.outputs

        outputs = self.sess.run(fetches, feed_dict)

        return outputs

    def get_local_variables(self):
        """
        Get all the variables that are local to the thread
        :return: a list of all the variables that are local to the thread
        """
        local_variables = [v for v in tf.local_variables() if self.online_network.name in v.name]
        if self.has_target:
            local_variables += [v for v in tf.local_variables() if self.target_network.name in v.name]
        return local_variables

    def get_global_variables(self):
        """
        Get all the variables that are shared between threads
        :return: a list of all the variables that are shared between threads
        """
        global_variables = [v for v in tf.global_variables() if self.global_network.name in v.name]
        return global_variables

    def set_session(self, sess):
        self.sess = sess
        self.online_network.set_session(sess)
        if self.global_network:
            self.global_network.set_session(sess)
        if self.target_network:
            self.target_network.set_session(sess)

