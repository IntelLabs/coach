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

import numpy as np
import copy
from configurations import *


class Memory:
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        pass

    def store(self, obj):
        pass

    def get(self, index):
        pass

    def length(self):
        pass

    def sample(self, size):
        pass

    def clean(self):
        pass


class Episode:
    def __init__(self):
        self.transitions = []
        # a num_transitions x num_transitions table with the n step return in the n'th row
        self.returns_table = None
        self._length = 0

    def insert(self, transition):
        self.transitions.append(transition)
        self._length += 1

    def is_empty(self):
        return self.length() == 0

    def length(self):
        return self._length

    def get_transition(self, transition_idx):
        return self.transitions[transition_idx]

    def get_last_transition(self):
        return self.get_transition(-1)

    def get_first_transition(self):
        return self.get_transition(0)

    def update_returns(self, discount, is_bootstrapped=False, n_step_return=-1):
        if n_step_return == -1 or n_step_return > self.length():
            n_step_return = self.length()
        rewards = np.array([t.reward for t in self.transitions])
        total_return = rewards.copy()
        current_discount = discount
        for i in range(1, n_step_return):
            total_return += current_discount * np.pad(rewards[i:], (0, i), 'constant', constant_values=0)
            current_discount *= discount

        if is_bootstrapped:
            bootstraps = np.array([np.squeeze(t.info['action_value']) for t in self.transitions[n_step_return:]])
            total_return += current_discount * np.pad(bootstraps, (0, n_step_return), 'constant', constant_values=0)

        for transition_idx in range(self.length()):
            self.transitions[transition_idx].total_return = total_return[transition_idx]

    def update_measurements_targets(self, num_steps):
        if 'measurements' not in self.transitions[0].state:
            return
        measurements_size = self.transitions[0].state['measurements'].shape[-1]
        total_return = sum([transition.reward for transition in self.transitions])
        for transition_idx, transition in enumerate(self.transitions):
            transition.info['future_measurements'] = np.zeros((num_steps, measurements_size))
            for step in range(num_steps):
                offset_idx = transition_idx + 2 ** step
                if offset_idx >= self.length():
                    offset_idx = -1
                transition.info['future_measurements'][step] = self.transitions[offset_idx].next_state['measurements'] - \
                                                               transition.state['measurements']
            transition.info['total_episode_return'] = total_return

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
        return [t.total_return for t in self.transitions]

    def to_batch(self):
        batch = []
        for i in range(self.length()):
            batch.append(self.get_transition(i))
        return batch


class Transition:
    def __init__(self, state, action, reward, next_state, game_over):
        self.state = copy.deepcopy(state)
        self.state['observation'] = np.array(self.state['observation'], copy=False)
        self.action = action
        self.reward = reward
        self.total_return = None
        self.next_state = copy.deepcopy(next_state)
        self.next_state['observation'] = np.array(self.next_state['observation'], copy=False)
        self.game_over = game_over
        self.info = {}
