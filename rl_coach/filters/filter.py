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
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Union, List

from rl_coach.core_types import EnvResponse, ActionInfo, Transition
from rl_coach.spaces import ActionSpace, RewardSpace, ObservationSpace
from rl_coach.utils import force_list


class Filter(object):
    def __init__(self, name=None):
        self.name = name

    def reset(self) -> None:
        """
        Called from reset() and implements the reset logic for the filter.
        :param name: the filter's name
        :return: None
        """
        pass

    def filter(self, env_response: Union[EnvResponse, Transition], update_internal_state: bool=True) \
            -> Union[EnvResponse, Transition]:
        """
        Filter some values in the env and return the filtered env_response
        This is the function that each filter should update
        :param update_internal_state: should the filter's internal state change due to this call
        :param env_response: the input env_response
        :return: the filtered env_response
        """
        raise NotImplementedError("")

    def set_device(self, device, memory_backend_params=None, mode='numpy') -> None:
        """
        An optional function that allows the filter to get the device if it is required to use tensorflow ops
        :param device: the device to use
        :param memory_backend_params: parameters associated with the memory backend
        :param mode: arithmetic backend to be used {numpy | tf}
        :return: None
        """
        pass

    def set_session(self, sess) -> None:
        """
        An optional function that allows the filter to get the session if it is required to use tensorflow ops
        :param sess: the session
        :return: None
        """
        pass

    def save_state_to_checkpoint(self, checkpoint_dir, checkpoint_prefix)->None:
        """
        Save the filter's internal state to a checkpoint to file, so that it can be later restored.
        :param checkpoint_dir: the directory in which to save the filter's state
        :param checkpoint_prefix: the prefix of the checkpoint file to save
        :return: None
        """
        pass

    def restore_state_from_checkpoint(self, checkpoint_dir, checkpoint_prefix)->None:
        """
        Save the filter's internal state to a checkpoint to file, so that it can be later restored.
        :param checkpoint_dir: the directory from which to restore
        :param checkpoint_prefix: the checkpoint prefix to look for
        :return: None
        """
        pass

    def set_name(self, name: str) -> None:
        """
        Set the filter's name
        :param name: the filter's name
        :return: None
        """
        self.name = name


class OutputFilter(Filter):
    """
    An output filter is a module that filters the output from an agent to the environment.
    """
    def __init__(self, action_filters: OrderedDict([(str, 'ActionFilter')])=None,
                 is_a_reference_filter: bool=False, name=None):
        super().__init__(name)

        if action_filters is None:
            action_filters = OrderedDict([])
        self._action_filters = action_filters

        # We do not want to allow reference filters such as Atari to be used directly. These have to be duplicated first
        # and only then can change their values so to keep their original params intact for other agents in the graph.
        self.i_am_a_reference_filter = is_a_reference_filter

    def __call__(self):
        duplicate = deepcopy(self)
        duplicate.i_am_a_reference_filter = False
        return duplicate

    def set_device(self, device, memory_backend_params=None, mode='numpy') -> None:
        """
        An optional function that allows the filter to get the device if it is required to use tensorflow ops
        :param device: the device to use
        :return: None
        """
        [f.set_device(device, memory_backend_params, mode='numpy') for f in self.action_filters.values()]

    def set_session(self, sess) -> None:
        """
        An optional function that allows the filter to get the session if it is required to use tensorflow ops
        :param sess: the session
        :return: None
        """
        [f.set_session(sess) for f in self.action_filters.values()]

    def filter(self, action_info: ActionInfo) -> ActionInfo:
        """
        A wrapper around _filter which first copies the action_info so that we don't change the original one
        This function should not be updated!
        :param action_info: the input action_info
        :return: the filtered action_info
        """
        if self.i_am_a_reference_filter:
            raise Exception("The filter being used is a reference filter. It is not to be used directly. "
                            "Instead get a duplicate from it by calling __call__.")
        if len(self.action_filters.values()) == 0:
            return action_info
        filtered_action_info = copy.deepcopy(action_info)
        filtered_action = filtered_action_info.action
        for filter in reversed(self.action_filters.values()):
            filtered_action = filter.filter(filtered_action)

        filtered_action_info.action = filtered_action

        return filtered_action_info

    def reverse_filter(self, action_info: ActionInfo) -> ActionInfo:
        """
        A wrapper around _reverse_filter which first copies the action_info so that we don't change the original one
        This function should not be updated!
        :param action_info: the input action_info
        :return: the filtered action_info
        """
        if self.i_am_a_reference_filter:
            raise Exception("The filter being used is a reference filter. It is not to be used directly. "
                            "Instead get a duplicate from it by calling __call__.")
        filtered_action_info = copy.deepcopy(action_info)
        filtered_action = filtered_action_info.action
        for filter in self.action_filters.values():
            filter.validate_output_action(filtered_action)
            filtered_action = filter.reverse_filter(filtered_action)

        filtered_action_info.action = filtered_action

        return filtered_action_info

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> ActionSpace:
        """
        Given the output action space, returns the corresponding unfiltered action space
        This function should not be updated!
        :param output_action_space: the output action space
        :return: the unfiltered action space
        """
        unfiltered_action_space = copy.deepcopy(output_action_space)
        for filter in self._action_filters.values():
            new_unfiltered_action_space = filter.get_unfiltered_action_space(unfiltered_action_space)
            filter.validate_output_action_space(unfiltered_action_space)
            unfiltered_action_space = new_unfiltered_action_space
        return unfiltered_action_space

    def reset(self) -> None:
        """
        Reset any internal memory stored in the filter.
        This function should not be updated!
        This is useful for stateful filters which stores information on previous filter calls.
        :return: None
        """
        [action_filter.reset() for action_filter in self._action_filters.values()]

    @property
    def action_filters(self) -> OrderedDict([(str, 'ActionFilter')]):
        return self._action_filters

    @action_filters.setter
    def action_filters(self, val: OrderedDict([(str, 'ActionFilter')])):
        self._action_filters = val

    def add_action_filter(self, filter_name: str, filter: 'ActionFilter', add_as_the_first_filter: bool=False):
        """
        Add an action filter to the filters list
        :param filter_name: the filter name
        :param filter: the filter to add
        :param add_as_the_first_filter: add the filter to the top of the filters stack
        :return: None
        """
        self._action_filters[filter_name] = filter
        if add_as_the_first_filter:
            self._action_filters.move_to_end(filter_name, last=False)

    def remove_action_filter(self, filter_name: str) -> None:
        """
        Remove an action filter from the filters list
        :param filter_name: the filter name
        :return: None
        """
        del self._action_filters[filter_name]

    def save_state_to_checkpoint(self, checkpoint_dir, checkpoint_prefix):
        """
        Currently not in use for OutputFilter.
        :param checkpoint_dir: the directory in which to save the filter's state
        :param checkpoint_prefix: the prefix of the checkpoint file to save
        :return:
        """
        pass

    def restore_state_from_checkpoint(self, checkpoint_dir, checkpoint_prefix)->None:
        """
        Currently not in use for OutputFilter.
        :param checkpoint_dir: the directory from which to restore
        :param checkpoint_prefix: the checkpoint prefix to look for
        :return: None
        """
        pass



class NoOutputFilter(OutputFilter):
    """
    Creates an empty output filter. Used only for readability when creating the presets
    """
    def __init__(self):
        super().__init__(is_a_reference_filter=False)


class InputFilter(Filter):
    """
    An input filter is a module that filters the input from an environment to the agent.
    """
    def __init__(self, observation_filters: Dict[str, Dict[str, 'ObservationFilter']]=None,
                 reward_filters: Dict[str, 'RewardFilter']=None,
                 is_a_reference_filter: bool=False, name=None):
        super().__init__(name)
        if observation_filters is None:
            observation_filters = {}
        if reward_filters is None:
            reward_filters = OrderedDict([])
        self._observation_filters = observation_filters
        self._reward_filters = reward_filters

        # We do not want to allow reference filters such as Atari to be used directly. These have to be duplicated first
        # and only then can change their values so to keep their original params intact for other agents in the graph.
        self.i_am_a_reference_filter = is_a_reference_filter

    def __call__(self):
        duplicate = deepcopy(self)
        duplicate.i_am_a_reference_filter = False
        return duplicate

    def set_device(self, device, memory_backend_params=None, mode='numpy') -> None:
        """
        An optional function that allows the filter to get the device if it is required to use tensorflow ops
        :param device: the device to use
        :return: None
        """
        [f.set_device(device, memory_backend_params, mode) for f in self.reward_filters.values()]
        [[f.set_device(device, memory_backend_params, mode) for f in filters.values()] for filters in self.observation_filters.values()]

    def set_session(self, sess) -> None:
        """
        An optional function that allows the filter to get the session if it is required to use tensorflow ops
        :param sess: the session
        :return: None
        """
        [f.set_session(sess) for f in self.reward_filters.values()]
        [[f.set_session(sess) for f in filters.values()] for filters in self.observation_filters.values()]

    def filter(self, unfiltered_data: Union[EnvResponse, List[EnvResponse], Transition, List[Transition]],
               update_internal_state: bool=True, deep_copy: bool=True) -> Union[List[EnvResponse], List[Transition]]:
        """
        A wrapper around _filter which first copies the env_response so that we don't change the original one
        This function should not be updated!
        :param unfiltered_data: the input data
        :param update_internal_state: should the filter's internal state change due to this call
        :return: the filtered env_response
        """
        if self.i_am_a_reference_filter:
            raise Exception("The filter being used is a reference filter. It is not to be used directly. "
                            "Instead get a duplicate from it by calling __call__.")
        if deep_copy:
            filtered_data = copy.deepcopy(unfiltered_data)
        else:
            filtered_data = [copy.copy(t) for t in unfiltered_data]
        filtered_data = force_list(filtered_data)

        # TODO: implement observation space validation
        # filter observations
        if isinstance(filtered_data[0], Transition):
            state_objects_to_filter = [[f.state for f in filtered_data],
                                       [f.next_state for f in filtered_data]]
        elif isinstance(filtered_data[0], EnvResponse):
            state_objects_to_filter = [[f.next_state for f in filtered_data]]
        else:
            raise ValueError("unfiltered_data should be either of type EnvResponse or Transition. ")

        for state_object_list in state_objects_to_filter:
            for observation_name, filters in self._observation_filters.items():
                if observation_name in state_object_list[0].keys():
                    for filter in filters.values():
                        data_to_filter = [state_object[observation_name] for state_object in state_object_list]
                        if filter.supports_batching:
                            filtered_observations = filter.filter(
                                data_to_filter, update_internal_state=update_internal_state)
                        else:
                            filtered_observations = []
                            for data_point in data_to_filter:
                                filtered_observations.append(filter.filter(
                                    data_point, update_internal_state=update_internal_state))

                        for i, state_object in enumerate(state_object_list):
                            state_object[observation_name] = filtered_observations[i]

        # filter reward
        for f in filtered_data:
            filtered_reward = f.reward
            for filter in self._reward_filters.values():
                filtered_reward = filter.filter(filtered_reward, update_internal_state)
            f.reward = filtered_reward

        return filtered_data

    def get_filtered_observation_space(self, observation_name: str,
                                       input_observation_space: ObservationSpace) -> ObservationSpace:
        """
        Given the input observation space, returns the corresponding filtered observation space
        This function should not be updated!
        :param observation_name: the name of the observation to which we want to calculate the filtered space
        :param input_observation_space: the input observation space
        :return: the filtered observation space
        """
        filtered_observation_space = copy.deepcopy(input_observation_space)
        if observation_name in self._observation_filters.keys():
            for filter in self._observation_filters[observation_name].values():
                filter.validate_input_observation_space(filtered_observation_space)
                filtered_observation_space = filter.get_filtered_observation_space(filtered_observation_space)
        return filtered_observation_space

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        """
        Given the input reward space, returns the corresponding filtered reward space
        This function should not be updated!
        :param input_reward_space: the input reward space
        :return: the filtered reward space
        """
        filtered_reward_space = copy.deepcopy(input_reward_space)
        for filter in self._reward_filters.values():
            filtered_reward_space = filter.get_filtered_reward_space(filtered_reward_space)
        return filtered_reward_space

    def reset(self) -> None:
        """
        Reset any internal memory stored in the filter.
        This function should not be updated!
        This is useful for stateful filters which stores information on previous filter calls.
        :return: None
        """
        for curr_observation_filters in self._observation_filters.values():
            [observation_filter.reset() for observation_filter in curr_observation_filters.values()]
        [reward_filter.reset() for reward_filter in self._reward_filters.values()]

    @property
    def observation_filters(self) -> Dict[str, Dict[str, 'ObservationFilter']]:
        return self._observation_filters

    @observation_filters.setter
    def observation_filters(self, val: Dict[str, Dict[str, 'ObservationFilter']]):
        self._observation_filters = val

    @property
    def reward_filters(self) -> OrderedDict([(str, 'RewardFilter')]):
        return self._reward_filters

    @reward_filters.setter
    def reward_filters(self, val: OrderedDict([(str, 'RewardFilter')])):
        self._reward_filters = val

    def copy_filters_from_one_observation_to_another(self, from_observation: str, to_observation: str):
        """
        Copy all the filters created for some observation to another observation
        :param from_observation: the source observation to copy from
        :param to_observation: the target observation to copy to
        :return: None
        """
        self._observation_filters[to_observation] = copy.deepcopy(self._observation_filters[from_observation])

    def add_observation_filter(self, observation_name: str, filter_name: str, filter: 'ObservationFilter',
                               add_as_the_first_filter: bool=False):
        """
        Add an observation filter to the filters list
        :param observation_name: the name of the observation to apply to
        :param filter_name: the filter name
        :param filter: the filter to add
        :param add_as_the_first_filter: add the filter to the top of the filters stack
        :return: None
        """
        if observation_name not in self._observation_filters.keys():
            self._observation_filters[observation_name] = OrderedDict()
        self._observation_filters[observation_name][filter_name] = filter
        if add_as_the_first_filter:
            self._observation_filters[observation_name].move_to_end(filter_name, last=False)

    def add_reward_filter(self, filter_name: str, filter: 'RewardFilter', add_as_the_first_filter: bool=False):
        """
        Add a reward filter to the filters list
        :param filter_name: the filter name
        :param filter: the filter to add
        :param add_as_the_first_filter: add the filter to the top of the filters stack
        :return: None
        """
        self._reward_filters[filter_name] = filter
        if add_as_the_first_filter:
            self._reward_filters.move_to_end(filter_name, last=False)

    def remove_observation_filter(self, observation_name: str, filter_name: str) -> None:
        """
        Remove an observation filter from the filters list
        :param observation_name: the name of the observation to apply to
        :param filter_name: the filter name
        :return: None
        """
        del self._observation_filters[observation_name][filter_name]

    def remove_reward_filter(self, filter_name: str) -> None:
        """
        Remove a reward filter from the filters list
        :param filter_name: the filter name
        :return: None
        """
        del self._reward_filters[filter_name]

    def save_state_to_checkpoint(self, checkpoint_dir, checkpoint_prefix):
        """
        Save the filter's internal state to a checkpoint to file, so that it can be later restored.
        :param checkpoint_dir: the directory in which to save the filter's state
        :param checkpoint_prefix: the prefix of the checkpoint file to save
        :return: None
        """
        checkpoint_prefix = '.'.join([checkpoint_prefix, 'filters'])
        if self.name is not None:
            checkpoint_prefix = '.'.join([checkpoint_prefix, self.name])
        for filter_name, filter in self._reward_filters.items():
            checkpoint_prefix = '.'.join([checkpoint_prefix, 'reward_filters', filter_name])
            filter.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)

        for observation_name, filters_dict in self._observation_filters.items():
            for filter_name, filter in filters_dict.items():
                checkpoint_prefix = '.'.join([checkpoint_prefix, 'observation_filters', observation_name,
                                                                 filter_name])
                filter.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)

    def restore_state_from_checkpoint(self, checkpoint_dir, checkpoint_prefix)->None:
        """
        Save the filter's internal state to a checkpoint to file, so that it can be later restored.
        :param checkpoint_dir: the directory from which to restore
        :param checkpoint_prefix: the checkpoint prefix to look for
        :return: None
        """
        checkpoint_prefix = '.'.join([checkpoint_prefix, 'filters'])
        if self.name is not None:
            checkpoint_prefix = '.'.join([checkpoint_prefix, self.name])
        for filter_name, filter in self._reward_filters.items():
            checkpoint_prefix = '.'.join([checkpoint_prefix, 'reward_filters', filter_name])
            filter.restore_state_from_checkpoint(checkpoint_dir, checkpoint_prefix)

        for observation_name, filters_dict in self._observation_filters.items():
            for filter_name, filter in filters_dict.items():
                checkpoint_prefix = '.'.join([checkpoint_prefix, 'observation_filters', observation_name,
                                                                 filter_name])
                filter.restore_state_from_checkpoint(checkpoint_dir, checkpoint_prefix)


class NoInputFilter(InputFilter):
    """
    Creates an empty input filter. Used only for readability when creating the presets
    """
    def __init__(self):
        super().__init__(is_a_reference_filter=False, name='no_input_filter')


