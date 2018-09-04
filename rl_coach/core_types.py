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
from enum import Enum
from random import shuffle
from typing import List, Union, Dict, Any, Type

import numpy as np

ActionType = Union[int, float, np.ndarray, List]
GoalType = Union[None, np.ndarray]
ObservationType = np.ndarray
RewardType = Union[int, float, np.ndarray]
StateType = Dict[str, np.ndarray]


class GoalTypes(Enum):
    Embedding = 1
    EmbeddingChange = 2
    Observation = 3
    Measurements = 4


# step methods

class StepMethod(object):
    def __init__(self, num_steps: int):
        self._num_steps = self.num_steps = num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @num_steps.setter
    def num_steps(self, val: int) -> None:
        self._num_steps = val


class Frames(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class EnvironmentSteps(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class EnvironmentEpisodes(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class TrainingSteps(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class Time(StepMethod):
    def __init__(self, num_steps):
        super().__init__(num_steps)


class PredictionType(object):
    pass


class VStateValue(PredictionType):
    pass


class QActionStateValue(PredictionType):
    pass


class ActionProbabilities(PredictionType):
    pass


class Embedding(PredictionType):
    pass


class InputEmbedding(Embedding):
    pass


class MiddlewareEmbedding(Embedding):
    pass


class InputImageEmbedding(InputEmbedding):
    pass


class InputVectorEmbedding(InputEmbedding):
    pass


class Middleware_FC_Embedding(MiddlewareEmbedding):
    pass


class Middleware_LSTM_Embedding(MiddlewareEmbedding):
    pass


class Measurements(PredictionType):
    pass

PlayingStepsType = Union[EnvironmentSteps, EnvironmentEpisodes, Frames]


# run phases
class RunPhase(Enum):
    HEATUP = "Heatup"
    TRAIN = "Training"
    TEST = "Testing"
    UNDEFINED = "Undefined"


# transitions

class Transition(object):
    def __init__(self, state: Dict[str, np.ndarray]=None, action: ActionType=None, reward: RewardType=None,
                 next_state: Dict[str, np.ndarray]=None, game_over: bool=None, info: Dict=None):
        """
        A transition is a tuple containing the information of a single step of interaction
        between the agent and the environment. The most basic version should contain the following values:
        (current state, action, reward, next state, game over)
        For imitation learning algorithms, if the reward, next state or game over is not known,
        it is sufficient to store the current state and action taken by the expert.

        :param state: The current state. Assumed to be a dictionary where the observation
                      is located at state['observation']
        :param action: The current action that was taken
        :param reward: The reward received from the environment
        :param next_state: The next state of the environment after applying the action.
                           The next state should be similar to the state in its structure.
        :param game_over: A boolean which should be True if the episode terminated after
                          the execution of the action.
        :param info: A dictionary containing any additional information to be stored in the transition
        """

        self._state = self.state = state
        self._action = self.action = action
        self._reward = self.reward = reward
        self._total_return = self.total_return = None
        if not next_state:
            next_state = state
        self._next_state = self._next_state = next_state
        self._game_over = self.game_over = game_over
        if info is None:
            self.info = {}
        else:
            self.info = info

    def __repr__(self):
        return str(self.__dict__)

    @property
    def state(self):
        if self._state is None:
            raise Exception("The state was not filled by any of the modules between the environment and the agent")
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def action(self):
        if self._action is None:
            raise Exception("The action was not filled by any of the modules between the environment and the agent")
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    @property
    def reward(self):

        if self._reward is None:
            raise Exception("The reward was not filled by any of the modules between the environment and the agent")
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def total_return(self):
        if self._total_return is None:
            raise Exception("The total_return was not filled by any of the modules between the environment and the "
                            "agent.  Make sure that you are using an episodic experience replay.")
        return self._total_return

    @total_return.setter
    def total_return(self, val):
        self._total_return = val

    @property
    def game_over(self):
        if self._game_over is None:
            raise Exception("The done flag was not filled by any of the modules between the environment and the agent")
        return self._game_over

    @game_over.setter
    def game_over(self, val):
        self._game_over = val

    @property
    def next_state(self):
        if self._next_state is None:
            raise Exception("The next state was not filled by any of the modules between the environment and the agent")
        return self._next_state

    @next_state.setter
    def next_state(self, val):
        self._next_state = val

    def add_info(self, new_info: Dict[str, Any]) -> None:
        if not new_info.keys().isdisjoint(self.info.keys()):
            raise ValueError("The new info dictionary can not be appended to the existing info dictionary since there "
                             "are overlapping keys between the two. old keys: {}, new keys: {}"
                             .format(self.info.keys(), new_info.keys()))
        self.info.update(new_info)

    def __copy__(self):
        new_transition = type(self)()
        new_transition.__dict__.update(self.__dict__)
        new_transition.state = copy.copy(new_transition.state)
        new_transition.next_state = copy.copy(new_transition.next_state)
        new_transition.info = copy.copy(new_transition.info)
        return new_transition


class EnvResponse(object):
    def __init__(self, next_state: Dict[str, ObservationType], reward: RewardType, game_over: bool, info: Dict=None,
                 goal: ObservationType=None):
        """
        An env response is a collection containing the information returning from the environment after a single action
        has been performed on it.
        :param next_state: The new state that the environment has transitioned into. Assumed to be a dictionary where the
                          observation is located at state['observation']
        :param reward: The reward received from the environment
        :param game_over: A boolean which should be True if the episode terminated after
                          the execution of the action.
        :param info: any additional info from the environment
        :param goal: a goal defined by the environment
        """
        self._next_state = self.next_state = next_state
        self._reward = self.reward = reward
        self._game_over = self.game_over = game_over
        self._goal = self.goal = goal
        if info is None:
            self.info = {}
        else:
            self.info = info

    def __repr__(self):
        return str(self.__dict__)

    @property
    def next_state(self):
        return self._next_state

    @next_state.setter
    def next_state(self, val):
        self._next_state = val

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def game_over(self):
        return self._game_over

    @game_over.setter
    def game_over(self, val):
        self._game_over = val

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, val):
        self._goal = val

    def add_info(self, info: Dict[str, Any]) -> None:
        if info.keys().isdisjoint(self.info.keys()):
            raise ValueError("The new info dictionary can not be appended to the existing info dictionary since there"
                             "are overlapping keys between the two")
        self.info.update(info)


class ActionInfo(object):
    """
    Action info is a class that holds an action and various additional information details about it
    """
    def __init__(self, action: ActionType, action_probability: float=0,
                 action_value: float=0., state_value: float=0., max_action_value: float=None,
                 action_intrinsic_reward: float=0):
        """
        :param action: the action
        :param action_probability: the probability that the action was given when selecting it
        :param action_value: the state-action value (Q value) of the action
        :param state_value: the state value (V value) of the state where the action was taken
        :param max_action_value: in case this is an action that was selected randomly, this is the value of the action
                                 that received the maximum value. if no value is given, the action is assumed to be the
                                 action with the maximum value
        :param action_intrinsic_reward: can contain any intrinsic reward that the agent wants to add to this action
                                        selection
        """
        self.action = action
        self.action_probability = action_probability
        self.action_value = action_value
        self.state_value = state_value
        if not max_action_value:
            self.max_action_value = action_value
        else:
            self.max_action_value = max_action_value
        self.action_intrinsic_reward = action_intrinsic_reward


class Batch(object):
    def __init__(self, transitions: List[Transition]):
        """
        A wrapper around a list of transitions that helps extracting batches of parameters from it.
        For example, one can extract a list of states corresponding to the list of transitions.
        The class uses lazy evaluation in order to return each of the available parameters.
        :param transitions: a list of transitions to extract the batch from
        """
        self.transitions = transitions
        self._states = {}
        self._actions = None
        self._rewards = None
        self._total_returns = None
        self._game_overs = None
        self._next_states = {}
        self._goals = None
        self._info = {}

    def slice(self, start, end) -> None:
        """
        Keep a slice from the batch and discard the rest of the batch
        :param start: the start index in the slice
        :param end: the end index in the slice
        :return: None
        """

        self.transitions = self.transitions[start:end]
        for k, v in self._states.items():
            self._states[k] = v[start:end]
        if self._actions is not None:
            self._actions = self._actions[start:end]
        if self._rewards is not None:
            self._rewards = self._rewards[start:end]
        if self._total_returns is not None:
            self._total_returns = self._total_returns[start:end]
        if self._game_overs is not None:
            self._game_overs = self._game_overs[start:end]
        for k, v in self._next_states.items():
            self._next_states[k] = v[start:end]
        if self._goals is not None:
            self._goals = self._goals[start:end]
        for k, v in self._info.items():
            self._info[k] = v[start:end]

    def shuffle(self) -> None:
        """
        Shuffle all the transitions in the batch
        :return: None
        """
        batch_order = list(range(self.size))
        shuffle(batch_order)
        self.transitions = [self.transitions[i] for i in batch_order]
        self._states = {}
        self._actions = None
        self._rewards = None
        self._total_returns = None
        self._game_overs = None
        self._next_states = {}
        self._goals = None
        self._info = {}

        # This seems to be slower
        # for k, v in self._states.items():
        #     self._states[k] = [v[i] for i in batch_order]
        # if self._actions is not None:
        #     self._actions = [self._actions[i] for i in batch_order]
        # if self._rewards is not None:
        #     self._rewards = [self._rewards[i] for i in batch_order]
        # if self._total_returns is not None:
        #     self._total_returns = [self._total_returns[i] for i in batch_order]
        # if self._game_overs is not None:
        #     self._game_overs = [self._game_overs[i] for i in batch_order]
        # for k, v in self._next_states.items():
        #     self._next_states[k] = [v[i] for i in batch_order]
        # if self._goals is not None:
        #     self._goals = [self._goals[i] for i in batch_order]
        # for k, v in self._info.items():
        #     self._info[k] = [v[i] for i in batch_order]

    def states(self, fetches: List[str], expand_dims=False) -> Dict[str, np.ndarray]:
        """
        follow the keys in fetches to extract the corresponding items from the states in the batch
        if these keys were not already extracted before. return only the values corresponding to those keys
        :param fetches: the keys of the state dictionary to extract
        :param expand_dims: add an extra dimension to each of the value batches
        :return: a dictionary containing a batch of values correponding to each of the given fetches keys
        """
        current_states = {}
        # there are cases (e.g. ddpg) where the state does not contain all the information needed for running
        # through the network and this has to be added externally (e.g. ddpg where the action needs to be given in
        # addition to the current_state, so that all the inputs of the network will be filled)
        for key in set(fetches).intersection(self.transitions[0].state.keys()):
            if key not in self._states.keys():
                self._states[key] = np.array([np.array(transition.state[key]) for transition in self.transitions])
            if expand_dims:
                current_states[key] = np.expand_dims(self._states[key], -1)
            else:
                current_states[key] = self._states[key]
        return current_states

    def actions(self, expand_dims=False) -> np.ndarray:
        """
        if the actions were not converted to a batch before, extract them to a batch and then return the batch
        :param expand_dims: add an extra dimension to the actions batch
        :return: a numpy array containing all the actions of the batch
        """
        if self._actions is None:
            self._actions = np.array([transition.action for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._actions, -1)
        return self._actions

    def rewards(self, expand_dims=False) -> np.ndarray:
        """
        if the rewards were not converted to a batch before, extract them to a batch and then return the batch
        :param expand_dims: add an extra dimension to the rewards batch
        :return: a numpy array containing all the rewards of the batch
        """
        if self._rewards is None:
            self._rewards = np.array([transition.reward for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._rewards, -1)
        return self._rewards

    def total_returns(self, expand_dims=False) -> np.ndarray:
        """
        if the total_returns were not converted to a batch before, extract them to a batch and then return the batch
        if the total return was not filled, this will raise an exception
        :param expand_dims: add an extra dimension to the total_returns batch
        :return: a numpy array containing all the total return values of the batch
        """
        if self._total_returns is None:
            self._total_returns = np.array([transition.total_return for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._total_returns, -1)
        return self._total_returns

    def game_overs(self, expand_dims=False) -> np.ndarray:
        """
        if the game_overs were not converted to a batch before, extract them to a batch and then return the batch
        :param expand_dims: add an extra dimension to the game_overs batch
        :return: a numpy array containing all the game over flags of the batch
        """
        if self._game_overs is None:
            self._game_overs = np.array([transition.game_over for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._game_overs, -1)
        return self._game_overs

    def next_states(self, fetches: List[str], expand_dims=False) -> Dict[str, np.ndarray]:
        """
        follow the keys in fetches to extract the corresponding items from the next states in the batch
        if these keys were not already extracted before. return only the values corresponding to those keys
        :param fetches: the keys of the state dictionary to extract
        :param expand_dims: add an extra dimension to each of the value batches
        :return: a dictionary containing a batch of values correponding to each of the given fetches keys
        """
        next_states = {}
        # there are cases (e.g. ddpg) where the state does not contain all the information needed for running
        # through the network and this has to be added externally (e.g. ddpg where the action needs to be given in
        # addition to the current_state, so that all the inputs of the network will be filled)
        for key in set(fetches).intersection(self.transitions[0].next_state.keys()):
            if key not in self._next_states.keys():
                self._next_states[key] = np.array([np.array(transition.next_state[key]) for transition in self.transitions])
            if expand_dims:
                next_states[key] = np.expand_dims(self._next_states[key], -1)
            else:
                next_states[key] = self._next_states[key]
        return next_states

    def goals(self, expand_dims=False) -> np.ndarray:
        """
        if the goals were not converted to a batch before, extract them to a batch and then return the batch
        if the goal was not filled, this will raise an exception
        :param expand_dims: add an extra dimension to the goals batch
        :return: a numpy array containing all the goals of the batch
        """
        if self._goals is None:
            self._goals = np.array([transition.goal for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._goals, -1)
        return self._goals

    def info(self, key, expand_dims=False) -> np.ndarray:
        """
        if the given info dictionary key was not converted to a batch before, extract it to a batch and then return the
        batch. if the key is not part of the keys in the info dictionary, this will raise an exception
        :param expand_dims: add an extra dimension to the info batch
        :return: a numpy array containing all the info values of the batch corresponding to the given key
        """
        if key not in self._info.keys():
            self._info[key] = np.array([transition.info[key] for transition in self.transitions])
        if expand_dims:
            return np.expand_dims(self._info[key], -1)
        return self._info[key]

    @property
    def size(self) -> int:
        """
        :return: the size of the batch
        """
        return len(self.transitions)

    def __getitem__(self, key):
        """
        get an item from the transitions list
        :param key: index of the transition in the batch
        :return: the transition corresponding to the given index
        """
        return self.transitions[key]

    def __setitem__(self, key, item):
        """
        set an item in the transition list
        :param key: index of the transition in the batch
        :param item: the transition to place in the given index
        :return: None
        """
        self.transitions[key] = item


class TotalStepsCounter(object):
    """
    A wrapper around a dictionary counting different StepMethods steps done.
    """
    def __init__(self):
        self.counters = {
            EnvironmentEpisodes: 0,
            EnvironmentSteps: 0,
            TrainingSteps: 0
        }

    def __getitem__(self, key: Type[StepMethod]) -> int:
        """
        get counter value
        :param key: counter type
        :return: the counter value
        """
        return self.counters[key]

    def __setitem__(self, key: StepMethod, item: int) -> None:
        """
        set an item in the transition list
        :param key: counter type
        :param item: an integer representing the new counter value
        :return: None
        """
        self.counters[key] = item


class GradientClippingMethod(Enum):
    ClipByGlobalNorm = 0
    ClipByNorm = 1
    ClipByValue = 2


class Episode(object):
    def __init__(self, discount: float=0.99, bootstrap_total_return_from_old_policy: bool=False, n_step: int=-1):
        """
        :param discount: the discount factor to use when calculating total returns
        :param bootstrap_total_return_from_old_policy: should the total return be bootstrapped from the values in the
                                                       memory
        :param n_step: the number of future steps to sum the reward over before bootstrapping
        """
        self.transitions = []
        # a num_transitions x num_transitions table with the n step return in the n'th row
        self.returns_table = None
        self._length = 0
        self.discount = discount
        self.bootstrap_total_return_from_old_policy = bootstrap_total_return_from_old_policy
        self.n_step = n_step
        self.is_complete = False

    def insert(self, transition):
        self.transitions.append(transition)
        self._length += 1
        if transition.game_over:
            self.is_complete = True

    def is_empty(self):
        return self.length() == 0

    def length(self):
        return self._length

    def get_transition(self, transition_idx):
        return self.transitions[transition_idx]

    def get_last_transition(self):
        return self.get_transition(-1) if self.length() > 0 else None

    def get_first_transition(self):
        return self.get_transition(0) if self.length() > 0 else None

    def update_returns(self):
        if self.n_step == -1 or self.n_step > self.length():
            curr_n_step = self.length()
        else:
            curr_n_step = self.n_step
        rewards = np.array([t.reward for t in self.transitions])
        rewards = rewards.astype('float')
        total_return = rewards.copy()
        current_discount = self.discount
        for i in range(1, curr_n_step):
            total_return += current_discount * np.pad(rewards[i:], (0, i), 'constant', constant_values=0)
            current_discount *= self.discount

        # calculate the bootstrapped returns
        if self.bootstrap_total_return_from_old_policy:
            bootstraps = np.array([np.squeeze(t.info['max_action_value']) for t in self.transitions[curr_n_step:]])
            bootstrapped_return = total_return + current_discount * np.pad(bootstraps, (0, curr_n_step), 'constant',
                                                                           constant_values=0)
            total_return = bootstrapped_return

        for transition_idx in range(self.length()):
            self.transitions[transition_idx].total_return = total_return[transition_idx]

    def update_actions_probabilities(self):
        probability_product = 1
        for transition_idx, transition in enumerate(self.transitions):
            if 'action_probabilities' in transition.info.keys():
                probability_product *= transition.info['action_probabilities']
        for transition_idx, transition in enumerate(self.transitions):
            transition.info['probability_product'] = probability_product

    def get_returns_table(self):
        return self.returns_table

    def get_returns(self):
        return self.get_transitions_attribute('total_return')

    def get_transitions_attribute(self, attribute_name):
        if len(self.transitions) > 0 and hasattr(self.transitions[0], attribute_name):
            return [getattr(t, attribute_name) for t in self.transitions]
        elif len(self.transitions) == 0:
            return []
        else:
            raise ValueError("The transitions have no such attribute name")

    def to_batch(self):
        batch = []
        for i in range(self.length()):
            batch.append(self.get_transition(i))
        return batch

    def __getitem__(self, sliced):
        return self.transitions[sliced]
