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

from typing import Union, Dict

from rl_coach.core_types import ActionType, EnvResponse, RunPhase
from rl_coach.spaces import ActionSpace


class EnvironmentInterface(object):
    def __init__(self):
        self._phase = RunPhase.UNDEFINED

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the environment
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the environment
        :param val: the new phase
        :return: None
        """
        self._phase = val

    @property
    def action_space(self) -> Union[Dict[str, ActionSpace], ActionSpace]:
        """
        Get the action space of the environment (or of each of the agents wrapped in this environment.
        i.e. in the LevelManager case")
        :return: the action space
        """
        raise NotImplementedError("")

    def get_random_action(self) -> ActionType:
        """
        Get a random action from the environment action space
        :return: An action that follows the definition of the action space.
        """
        raise NotImplementedError("")

    def step(self, action: ActionType) -> Union[None, EnvResponse]:
        """
        Make a single step in the environment using the given action
        :param action: an action to use for stepping the environment. Should follow the definition of the action space.
        :return: the environment response as returned in get_last_env_response or None for LevelManager
        """
        raise NotImplementedError("")

    def reset_internal_state(self, force_environment_reset: bool=False) -> Union[None, EnvResponse]:
        """
        Reset the environment episode
        :param force_environment_reset: in some cases, resetting the environment can be suppressed by the environment
                                        itself. This flag allows force the reset.
        :return: the environment response as returned in get_last_env_response or None for LevelManager
        """
        raise NotImplementedError("")
