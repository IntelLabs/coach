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

from typing import List, Dict

import numpy as np

from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.layers import NoisyNetDense
from rl_coach.base_parameters import AgentParameters, NetworkParameters
from rl_coach.spaces import ActionSpace, BoxActionSpace, DiscreteActionSpace

from rl_coach.core_types import ActionType
from rl_coach.exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters


class ParameterNoiseParameters(ExplorationParameters):
    def __init__(self, agent_params: AgentParameters):
        super().__init__()
        if not isinstance(agent_params, DQNAgentParameters):
            raise ValueError("Currently only DQN variants are supported for using an exploration type of "
                             "ParameterNoise.")

        self.network_params = agent_params.network_wrappers

    @property
    def path(self):
        return 'rl_coach.exploration_policies.parameter_noise:ParameterNoise'


class ParameterNoise(ExplorationPolicy):
    """
    The ParameterNoise exploration policy is intended for both discrete and continuous action spaces.
    It applies the exploration policy by replacing all the dense network layers with noisy layers.
    The noisy layers have both weight means and weight standard deviations, and for each forward pass of the network
    the weights are sampled from a normal distribution that follows the learned weights mean and standard deviation
    values.

    Warning: currently supported only by DQN variants
    """
    def __init__(self, network_params: Dict[str, NetworkParameters], action_space: ActionSpace):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)
        self.network_params = network_params
        self._replace_network_dense_layers()

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if type(self.action_space) == DiscreteActionSpace:
            return np.argmax(action_values)
        elif type(self.action_space) == BoxActionSpace:
            action_values_mean = action_values[0].squeeze()
            action_values_std = action_values[1].squeeze()
            return np.random.normal(action_values_mean, action_values_std)
        else:
            raise ValueError("ActionSpace type {} is not supported for ParameterNoise.".format(type(self.action_space)))

    def get_control_param(self):
        return 0

    def _replace_network_dense_layers(self):
        # replace the dense type for all the networks components (embedders, mw, heads) with a NoisyNetDense

        # NOTE: we are changing network params in a non-params class (an already instantiated class), this could have
        #       been prone to a bug, but since the networks are created very late in the game
        #       (after agent.init_environment_dependent()_modules is called) - then we are fine.

        for network_wrapper_params in self.network_params.values():
            for component_params in list(network_wrapper_params.input_embedders_parameters.values()) + \
                                    [network_wrapper_params.middleware_parameters] + \
                                    network_wrapper_params.heads_parameters:
                component_params.dense_layer = NoisyNetDense

