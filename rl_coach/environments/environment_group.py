
########################################################################################################################
####### Currently we are ignoring more complex cases including EnvironmentGroups - DO NOT USE THIS FILE ****************
########################################################################################################################




# #
# # Copyright (c) 2017 Intel Corporation
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
# from typing import Union, List, Dict
# import numpy as np
# from environments import create_environment
# from environments.environment import Environment
# from environments.environment_interface import EnvironmentInterface, ActionType, ActionSpace
# from core_types import GoalType, Transition
#
#
# class EnvironmentGroup(EnvironmentInterface):
#     """
#     An EnvironmentGroup is a group of different environments.
#     In the simple case, it will contain a single environment. But it can also contain multiple environments,
#     where the agent can then act on them as a batch, such that the prediction of the action is more efficient.
#     """
#     def __init__(self, environments_parameters: List[Environment]):
#         self.environments_parameters = environments_parameters
#         self.environments = []
#         self.action_space = []
#         self.outgoing_control = []
#         self._last_env_response = []
#
#     @property
#     def action_space(self) -> Union[List[ActionSpace], ActionSpace]:
#         """
#         Get the action space of the environment
#         :return: the action space
#         """
#         return self.action_space
#
#     @action_space.setter
#     def action_space(self, val: Union[List[ActionSpace], ActionSpace]):
#         """
#         Set the action space of the environment
#         :return: None
#         """
#         self.action_space = val
#
#     @property
#     def phase(self) -> RunPhase:
#         """
#         Get the phase of the environments group
#         :return: the current phase
#         """
#         return self.phase
#
#     @phase.setter
#     def phase(self, val: RunPhase):
#         """
#         Change the phase of each one of the environments in the group
#         :param val: the new phase
#         :return: None
#         """
#         self.phase = val
#         call_method_for_all(self.environments, 'phase', val)
#
#     def _create_environments(self):
#         """
#         Create the environments using the given parameters and update the environments list
#         :return: None
#         """
#         for environment_parameters in self.environments_parameters:
#             environment = create_environment(environment_parameters)
#             self.action_space = self.action_space.append(environment.action_space)
#             self.environments.append(environment)
#
#    @property
#    def last_env_response(self) -> Union[List[Transition], Transition]:
#        """
#        Get the last environment response
#        :return: a dictionary that contains the state, reward, etc.
#        """
#        return squeeze_list(self._last_env_response)
#
#    @last_env_response.setter
#    def last_env_response(self, val: Union[List[Transition], Transition]):
#        """
#        Set the last environment response
#        :param val: the last environment response
#        """
#        self._last_env_response = force_list(val)
#
#     def step(self, actions: Union[List[ActionType], ActionType]) -> List[Transition]:
#         """
#         Act in all the environments in the group.
#         :param actions: can be either a single action if there is a single environment in the group, or a list of
#                         actions in case there are multiple environments in the group. Each action can be an action index
#                         or a numpy array representing a continuous action for example.
#         :return: The responses from all the environments in the group
#         """
#
#         actions = force_list(actions)
#         if len(actions) != len(self.environments):
#             raise ValueError("The number of actions does not match the number of environments in the group")
#
#         result = []
#         for environment, action in zip(self.environments, actions):
#             result.append(environment.step(action))
#
#         self.last_env_response = result
#
#         return result
#
#     def reset(self, force_environment_reset: bool=False) -> List[Transition]:
#         """
#         Reset all the environments in the group
#         :param force_environment_reset: force the reset of each one of the environments
#         :return: a list of the environments responses
#         """
#         return call_method_for_all(self.environments, 'reset', force_environment_reset)
#
#     def get_random_action(self) -> List[ActionType]:
#        """
#        Get a list of random action that can be applied on the environments in the group
#        :return: a list of random actions
#        """
#         return call_method_for_all(self.environments, 'get_random_action')
#
#     def set_goal(self, goal: GoalType) -> None:
#         """
#         Set the goal of each one of the environments in the group to be the given goal
#         :param goal: a goal vector
#         :return: None
#         """
#         # TODO: maybe enable setting multiple goals?
#         call_method_for_all(self.environments, 'set_goal', goal)
