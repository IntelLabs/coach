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

import random
from enum import Enum
from itertools import product
from typing import Union, List, Dict, Tuple, Callable

import numpy as np
import scipy
import scipy.spatial

from rl_coach.core_types import ActionType, ActionInfo
from rl_coach.utils import eps


class Space(object):
    """
    A space defines a set of valid values
    """
    def __init__(self, shape: Union[int, tuple, list, np.ndarray], low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf):
        """
        :param shape: the shape of the space
        :param low: the lowest values possible in the space. can be an array defining the lowest values per point,
                    or a single value defining the general lowest values
        :param high: the highest values possible in the space. can be an array defining the highest values per point,
                    or a single value defining the general highest values
        """

        # the number of dimensions is the number of axes in the shape. it will be set in the shape setter
        self.num_dimensions = 0

        # the number of elements is the number of possible actions if the action space was discrete.
        # it will be set in the shape setter
        self.num_elements = 0

        self._low = self._high = None
        self._shape = self.shape = shape
        self._low = self.low = low
        self._high = self.high = high

        # we allow zero sized spaces which means that the space is empty. this is useful for environments with no
        # measurements for example.
        if type(shape) == int and shape < 0:
            raise ValueError("The shape of the space must be a non-negative number")

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val: Union[int, tuple, list, np.ndarray]):
        # convert the shape to an np.ndarray
        self._shape = val
        if type(self._shape) == int:
            self._shape = np.array([self._shape])
        if type(self._shape) == tuple or type(self._shape) == list:
            self._shape = np.array(self._shape)

        # the shape is now an np.ndarray
        self.num_dimensions = len(self._shape)
        self.num_elements = int(np.prod(self._shape))

    @property
    def low(self):
        if hasattr(self, '_low'):
            return self._low
        else:
            return None

    @low.setter
    def low(self, val: Union[None, int, float, np.ndarray]):
        if type(val) == np.ndarray and type(self.shape) == np.ndarray and np.all(val.shape != self.shape):
            raise ValueError("The low values shape don't match the shape of the space")
        elif self.high is not None and not np.all(self.high >= val):
            raise ValueError("At least one of the axes-parallel lines defining the space has high values which "
                             "are lower than the given low values")
        else:
            self._low = val
            # we allow using a number to define the low values, but we immediately convert it to an array which defines
            # the low values for all the space dimensions in order to expose a consistent value type
            if type(self._low) == int or type(self._low) == float:
                self._low = np.ones(self.shape)*self._low

    @property
    def high(self):
        if hasattr(self, '_high'):
            return self._high
        else:
            return None

    @high.setter
    def high(self, val: Union[None, int, float, np.ndarray]):
        if type(val) == np.ndarray and type(self.shape) == np.ndarray and np.all(val.shape != self.shape):
            raise ValueError("The high values shape don't match the shape of the space")
        elif self.low is not None and not np.all(self.low <= val):
            raise ValueError("At least one of the axes-parallel lines defining the space has low values which "
                             "are higher than the given high values")
        else:
            self._high = val
            # we allow using a number to define the high values, but we immediately convert it to an array which defines
            # the high values for all the space dimensions in order to expose a consistent value type
            if type(self._high) == int or type(self._high) == float:
                self._high = np.ones(self.shape)*self._high

    def val_matches_space_definition(self, val: Union[int, float, np.ndarray]) -> bool:
        """
        Checks if the given value matches the space definition in terms of shape and values

        :param val: a value to check
        :return: True / False depending on if the val matches the space definition
        """
        if (type(val) == int or type(val) == float) and not np.all(self.shape == np.ones(1)):
            return False
        if type(val) == np.ndarray and not np.all(val.shape == self.shape):
            return False
        if (self.low is not None and not np.all(val >= self.low)) \
                or (self.high is not None and not np.all(val <= self.high)):
            # TODO: check the performance overhead this causes
            return False
        return True

    def is_point_in_space_shape(self, point: np.ndarray) -> bool:
        """
        Checks if a given multidimensional point is within the bounds of the shape of the space

        :param point: a multidimensional point
        :return: True if the point is within the shape of the space. False otherwise
        """
        if len(point) != self.num_dimensions:
            return False
        if np.any(point < np.zeros(self.num_dimensions)) or np.any(point >= self.shape):
            return False
        return True

    def sample(self) -> np.ndarray:
        """
        Sample the defined space, either uniformly, if space bounds are defined, or Normal distributed if no
        bounds are defined

        :return: A numpy array sampled from the space
        """
        # if there are infinite bounds, we sample using gaussian noise with mean 0 and std 1
        if np.any(self.low == -np.inf) or np.any(self.high == np.inf):
            return np.random.normal(0, 1, self.shape)
        else:
            return np.random.uniform(self.low, self.high, self.shape)


class RewardSpace(Space):
    def __init__(self, shape: Union[int, np.ndarray], low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf,
                 reward_success_threshold: Union[None, int, float]=None):
        super().__init__(shape, low, high)
        self.reward_success_threshold = reward_success_threshold


"""
Observation Spaces
"""


class ObservationSpace(Space):
    def __init__(self, shape: Union[int, np.ndarray], low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf):
        super().__init__(shape, low, high)


class VectorObservationSpace(ObservationSpace):
    """
    An observation space which is defined as a vector of elements. This can be particularly useful for environments
    which return measurements, such as in robotic environments.
    """
    def __init__(self, shape: int, low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf, measurements_names: List[str]=None):
        if measurements_names is None:
            measurements_names = []
        if len(measurements_names) > shape:
            raise ValueError("measurement_names size {} is larger than shape {}.".format(
                len(measurements_names), shape))

        self.measurements_names = measurements_names
        super().__init__(shape, low, high)


class TensorObservationSpace(ObservationSpace):
    """
    An observation space which defines observations with arbitrary shape. This can be particularly useful for
    environments with non image input.
    """
    def __init__(self, shape: np.ndarray, low: -np.inf,
                 high: np.inf):
        super().__init__(shape, low, high)


class PlanarMapsObservationSpace(ObservationSpace):
    """
    An observation space which defines a stack of 2D observations. For example, an environment which returns
    a stack of segmentation maps like in Starcraft.
    """
    def __init__(self, shape: Union[np.ndarray], low: int, high: int, channels_axis: int=-1):
        super().__init__(shape, low, high)
        self.channels_axis = channels_axis

        if not 2 <= len(shape) <= 3:
            raise ValueError("Planar maps observations must have 3 dimensions - a channels dimension and 2 maps "
                             "dimensions, not {}".format(len(shape)))
        if len(shape) == 2:
            self.channels = 1
        else:
            self.channels = shape[channels_axis]


class ImageObservationSpace(PlanarMapsObservationSpace):
    """
    An observation space which is a private case of the PlanarMapsObservationSpace, where the stack of 2D observations
    represent a RGB image, or a grayscale image.
    """
    def __init__(self, shape: Union[np.ndarray], high: int, channels_axis: int=-1):
        # TODO: consider allowing arbitrary low values for images
        super().__init__(shape, 0, high, channels_axis)
        self.has_colors = self.channels == 3
        if not self.channels == 3 and not self.channels == 1:
            raise ValueError("Image observations must have 1 or 3 channels, not {}".format(self.channels))


# TODO: mixed observation spaces (image + measurements, image + segmentation + depth map, etc.)
class StateSpace(object):
    def __init__(self, sub_spaces: Dict[str, Space]):
        self.sub_spaces = sub_spaces

    def __getitem__(self, item):
        return self.sub_spaces[item]

    def __setitem__(self, key, value):
        self.sub_spaces[key] = value


"""
Action Spaces
"""


class ActionSpace(Space):
    def __init__(self, shape: Union[int, np.ndarray], low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf, descriptions: Union[None, List, Dict]=None,
                 default_action: ActionType=None):
        super().__init__(shape, low, high)
        # we allow a mismatch between the number of descriptions and the number of actions.
        # in this case the descriptions for the actions that were not given will be the action index
        if descriptions is not None:
            self.descriptions = descriptions
        else:
            self.descriptions = {}
        self.default_action = default_action

    @property
    def actions(self) -> List[ActionType]:
        raise NotImplementedError("The action space does not have an explicit actions list")

    def sample_with_info(self) -> ActionInfo:
        """
        Get a random action with additional "fake" info

        :return: An action info instance
        """
        return ActionInfo(self.sample())

    def clip_action_to_space(self, action: ActionType) -> ActionType:
        """
        Given an action, clip its values to fit to the action space ranges

        :param action: a given action
        :return: the clipped action
        """
        return action

    def get_description(self, action: np.ndarray) -> str:
        raise NotImplementedError("")

    def __str__(self):
        return "{}: shape = {}, low = {}, high = {}".format(self.__class__.__name__, self.shape, self.low, self.high)

    def __repr__(self):
        return self.__str__()


class AttentionActionSpace(ActionSpace):
    """
    A box selection continuous action space, meaning that the actions are defined as selecting a multidimensional box
    from a given range.
    The actions will be in the form:
    [[low_x, low_y, ...], [high_x, high_y, ...]]
    """
    def __init__(self, shape: int, low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf, descriptions: Union[None, List, Dict]=None,
                 default_action: np.ndarray = None, forced_attention_size: Union[None, int, float, np.ndarray]=None):
        super().__init__(shape, low, high, descriptions)

        self.forced_attention_size = forced_attention_size
        if isinstance(self.forced_attention_size, int) or isinstance(self.forced_attention_size, float):
            self.forced_attention_size = np.ones(self.shape) * self.forced_attention_size

        if self.forced_attention_size is not None and np.all(self.forced_attention_size > (self.high - self.low)):
            raise ValueError("The forced attention size is larger than the action space")

        # default action
        if default_action is None:
            if self.forced_attention_size is not None:
                self.default_action = [self.low*np.ones(self.shape),
                                       (self.low+self.forced_attention_size)*np.ones(self.shape)]
            else:
                self.default_action = [self.low*np.ones(self.shape), self.high*np.ones(self.shape)]
        else:
            self.default_action = default_action

    def sample(self) -> List:
        if self.forced_attention_size is not None:
            sampled_low = np.random.uniform(self.low, self.high-self.forced_attention_size, self.shape)
            sampled_high = sampled_low + self.forced_attention_size
        else:
            sampled_low = np.random.uniform(self.low, self.high, self.shape)
            sampled_high = np.random.uniform(sampled_low, self.high, self.shape)
        return [sampled_low, sampled_high]

    def clip_action_to_space(self, action: ActionType) -> ActionType:
        action = [np.clip(action[0], self.low, self.high), np.clip(action[1], self.low, self.high)]
        return action


class BoxActionSpace(ActionSpace):
    """
    A multidimensional bounded or unbounded continuous action space
    """
    def __init__(self, shape: Union[int, np.ndarray], low: Union[None, int, float, np.ndarray]=-np.inf,
                 high: Union[None, int, float, np.ndarray]=np.inf, descriptions: Union[None, List, Dict]=None,
                 default_action: np.ndarray=None):
        super().__init__(shape, low, high, descriptions)
        self.max_abs_range = np.maximum(np.abs(self.low), np.abs(self.high))

        # default action
        if default_action is None:
            if np.any(np.isinf(self.low)) or np.any(np.isinf(self.high)):
                self.default_action = np.zeros(shape)
            else:
                self.default_action = self.low + (self.high - self.low) / 2
        else:
            self.default_action = default_action

    def clip_action_to_space(self, action: ActionType) -> ActionType:
        action = np.clip(action, self.low, self.high)
        return action


class DiscreteActionSpace(ActionSpace):
    """
    A discrete action space with action indices as actions
    """
    def __init__(self, num_actions: int, descriptions: Union[None, List, Dict]=None, default_action: np.ndarray=None):
        super().__init__(1, low=0, high=num_actions-1, descriptions=descriptions)
        # the number of actions is mapped to high

        # default action
        if default_action is None:
            self.default_action = 0
        else:
            self.default_action = default_action

    @property
    def actions(self) -> List[ActionType]:
        return list(range(0, int(self.high[0]) + 1))

    def sample(self) -> int:
        return np.random.choice(self.actions)

    def sample_with_info(self) -> ActionInfo:
        return ActionInfo(self.sample(), action_probability=1. / (self.high[0] - self.low[0] + 1))

    def get_description(self, action: int) -> str:
        if type(self.descriptions) == list and 0 <= action < len(self.descriptions):
            return self.descriptions[action]
        elif type(self.descriptions) == dict and action in self.descriptions.keys():
            return self.descriptions[action]
        elif 0 <= action < self.shape:
            return str(action)
        else:
            raise ValueError("The given action is outside of the action space")


class MultiSelectActionSpace(ActionSpace):
    """
    A discrete action space where multiple actions can be selected at once. The actions are encoded as multi-hot vectors
    """
    def __init__(self, size: int, max_simultaneous_selected_actions: int=1, descriptions: Union[None, List, Dict]=None,
                 default_action: np.ndarray=None, allow_no_action_to_be_selected=True):
        super().__init__(size, low=None, high=None, descriptions=descriptions)
        self.max_simultaneous_selected_actions = max_simultaneous_selected_actions

        if max_simultaneous_selected_actions > size:
            raise ValueError("The maximum simultaneous selected actions can't be larger the max number of actions")

        # create all combinations of actions as a list of actions
        I = [np.eye(size)]*self.max_simultaneous_selected_actions
        self._actions = []
        if allow_no_action_to_be_selected:
            self._actions.append(np.zeros(size))
        self._actions.extend(list(np.unique([np.clip(np.sum(t, axis=0), 0, 1) for t in product(*I)], axis=0)))

        # default action
        if default_action is None:
            self.default_action = self._actions[0]
        else:
            self.default_action = default_action

    @property
    def actions(self) -> List[ActionType]:
        return self._actions

    def sample(self) -> np.ndarray:
        # samples a multi-hot vector
        return random.choice(self.actions)

    def sample_with_info(self) -> ActionInfo:
        return ActionInfo(self.sample(), action_probability=1. / len(self.actions))

    def get_description(self, action: np.ndarray) -> str:
        if np.sum(len(np.where(action == 0)[0])) + np.sum(len(np.where(action == 1)[0])) != self.shape or \
                        np.sum(len(np.where(action == 1)[0])) > self.max_simultaneous_selected_actions:
            raise ValueError("The given action is not in the action space")
        selected_actions = np.where(action == 1)[0]
        description = [self.descriptions[a] for a in selected_actions]
        if len(description) == 0:
            description = ['no-op']
        return ' + '.join(description)


class CompoundActionSpace(ActionSpace):
    """
    An action space which consists of multiple sub-action spaces.
    For example, in Starcraft the agent should choose an action identifier from ~550 options (Discrete(550)),
    but it also needs to choose 13 different arguments for the selected action identifier, where each argument is
    by itself an action space. In Starcraft, the arguments are Discrete action spaces as well, but this is not mandatory.
    """
    def __init__(self, sub_spaces: List[ActionSpace]):
        super().__init__(0)
        self.sub_action_spaces = sub_spaces
        # TODO: define the shape, low and high value in a better way

    @property
    def actions(self) -> List[ActionType]:
        return [action_space.actions for action_space in self.sub_action_spaces]

    def sample(self) -> ActionType:
        return [action_space.sample() for action_space in self.sub_action_spaces]

    def clip_action_to_space(self, actions: List[ActionType]) -> ActionType:
        if not isinstance(actions, list) or len(actions) != len(self.sub_action_spaces):
            raise ValueError("The actions to be clipped must be a list with the same number of sub-actions as "
                             "defined in the compound action space.")
        for idx in range(len(self.sub_action_spaces)):
            actions[idx] = self.sub_action_spaces[idx].clip_action_to_space(actions[idx])
        return actions

    def get_description(self, actions: np.ndarray) -> str:
        description = [action_space.get_description(action) for action_space, action in zip(self.sub_action_spaces, actions)]
        return ' + '.join(description)


"""
Goals
"""


class GoalToRewardConversion(object):
    def __init__(self, goal_reaching_reward: float=0):
        self.goal_reaching_reward = goal_reaching_reward

    def convert_distance_to_reward(self, distance: Union[float, np.ndarray]) -> Tuple[float, bool]:
        """
        Given a distance from the goal, return a reward and a flag representing if the goal was reached

        :param distance: the distance from the goal
        :return:
        """
        raise NotImplementedError("")


class ReachingGoal(GoalToRewardConversion):
    """
    get a reward if the goal was reached and 0 otherwise
    """
    def __init__(self, distance_from_goal_threshold: Union[float, np.ndarray], goal_reaching_reward: float=0,
                 default_reward: float=-1):
        """
        :param distance_from_goal_threshold: consider getting to this distance from the goal the same as getting
                                             to the goal
        :param goal_reaching_reward: the reward the agent will get when reaching the goal
        :param default_reward: the reward the agent will get until it reaches the goal
        """
        super().__init__(goal_reaching_reward)
        self.distance_from_goal_threshold = distance_from_goal_threshold
        self.default_reward = default_reward

    def convert_distance_to_reward(self, distance: Union[float, np.ndarray]) -> Tuple[float, bool]:
        if np.all(distance <= self.distance_from_goal_threshold):
            return self.goal_reaching_reward, True
        else:
            return self.default_reward, False


class InverseDistanceFromGoal(GoalToRewardConversion):
    """
    get a reward inversely proportional to the distance from the goal
    """
    def __init__(self, distance_from_goal_threshold: Union[float, np.ndarray], max_reward: float=1):
        """
        :param distance_from_goal_threshold: consider getting to this distance from the goal the same as getting
                                             to the goal
        :param max_reward: the max reward the agent can get
        """
        super().__init__(goal_reaching_reward=max_reward)
        self.distance_from_goal_threshold = distance_from_goal_threshold
        self.max_reward = max_reward

    def convert_distance_to_reward(self, distance: Union[float, np.ndarray]) -> Tuple[float, bool]:
        return min(self.max_reward, 1 / (distance + eps)), distance <= self.distance_from_goal_threshold


class GoalsSpace(VectorObservationSpace, ActionSpace):
    """
    A multidimensional space with a goal type definition. It also behaves as an action space, so that hierarchical
    agents can use it as an output action space.
    The class acts as a wrapper to the target space. So after setting the target space, all the values of the class
    will match the values of the target space (the shape, low, high, etc.)
    """
    class DistanceMetric(Enum):
        Euclidean = 0
        Cosine = 1
        Manhattan = 2

    def __init__(self, goal_name: str, reward_type: GoalToRewardConversion,
                 distance_metric: Union[DistanceMetric, Callable]):
        """
        :param goal_name: the name of the observation space to use as the achieved goal.
        :param reward_type: the reward type to use for converting distances from goal to rewards
        :param distance_metric: the distance metric to use. could be either one of the distances in the
                                DistanceMetric enum, or a custom function that gets two vectors as input and
                                returns the distance between them
        """
        super().__init__(0)
        self.goal_name = goal_name
        self.distance_metric = distance_metric
        self.reward_type = reward_type
        self.target_space = None
        self.max_abs_range = None

    def set_target_space(self, target_space: Space) -> None:
        self.target_space = target_space
        super().__init__(self.target_space.shape, self.target_space.low, self.target_space.high)
        self.max_abs_range = np.maximum(np.abs(self.low), np.abs(self.high))

    def goal_from_state(self, state: Dict):
        """
        Given a state, extract an observation according to the goal_name

        :param state: a dictionary of observations
        :return: the observation corresponding to the goal_name
        """
        return state[self.goal_name]

    def distance_from_goal(self, goal: np.ndarray, state: dict) -> float:
        """
        Given a state, check its distance from the goal

        :param goal: a numpy array representing the goal
        :param state: a dict representing the state
        :return: the distance from the goal
        """
        state_value = self.goal_from_state(state)

        # calculate distance
        if self.distance_metric == self.DistanceMetric.Cosine:
            dist = scipy.spatial.distance.cosine(goal, state_value)
        elif self.distance_metric == self.DistanceMetric.Euclidean:
            dist = scipy.spatial.distance.euclidean(goal, state_value)
        elif self.distance_metric == self.DistanceMetric.Manhattan:
            dist = scipy.spatial.distance.cityblock(goal, state_value)
        elif callable(self.distance_metric):
            dist = self.distance_metric(goal, state_value)
        else:
            raise ValueError("The given distance metric for the goal is not valid.")

        return dist

    def get_reward_for_goal_and_state(self, goal: np.ndarray, state: dict) -> Tuple[float, bool]:
        """
        Given a state, check if the goal was reached and return a reward accordingly

        :param goal: a numpy array representing the goal
        :param state: a dict representing the state
        :return: the reward for the current goal and state pair and a boolean representing if the goal was reached
        """
        dist = self.distance_from_goal(goal, state)
        return self.reward_type.convert_distance_to_reward(dist)


class AgentSelection(DiscreteActionSpace):
    """
    An discrete action space which is bounded by the number of agents to select from
    """
    def __init__(self, num_agents: int):
        super().__init__(num_agents)


class SpacesDefinition(object):
    """
    A container class that allows passing the definitions of all the spaces at once
    """
    def __init__(self,
                 state: StateSpace,
                 goal: ObservationSpace,
                 action: ActionSpace,
                 reward: RewardSpace):
        self.state = state
        self.goal = goal
        self.action = action
        self.reward = reward
