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

import operator
import random
from enum import Enum
from typing import List, Tuple, Any, Union

import numpy as np

from rl_coach.core_types import Transition
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters, ExperienceReplay
from rl_coach.schedules import Schedule, ConstantSchedule


class BalancedExperienceReplayParameters(ExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 1000000)
        self.allow_duplicates_in_batch_sampling = False
        self.num_classes = 0
        self.state_key_with_the_class_index = 'class'

    @property
    def path(self):
        return 'rl_coach.memories.non_episodic.balanced_experience_replay:BalancedExperienceReplay'


"""
A replay buffer which allows sampling batches which are balanced in terms of the classes that are sampled
"""
class BalancedExperienceReplay(ExperienceReplay):
    def __init__(self, max_size: Tuple[MemoryGranularity, int], allow_duplicates_in_batch_sampling: bool=True,
                 num_classes: int=0, state_key_with_the_class_index: Any='class'):
        """
        :param max_size: the maximum number of transitions or episodes to hold in the memory
        :param allow_duplicates_in_batch_sampling: allow having the same transition multiple times in a batch
        :param num_classes: the number of classes in the replayed data
        :param state_key_with_the_class_index: the class index is assumed to be a value in the state dictionary.
                                           this parameter determines the key to retrieve the class index value
        """
        super().__init__(max_size, allow_duplicates_in_batch_sampling)
        self.current_class_to_sample_from = 0
        self.num_classes = num_classes
        self.state_key_with_the_class_index = state_key_with_the_class_index
        self.transitions = [[] for _ in range(self.num_classes)]
        self.transitions_order = []

        if self.num_classes < 2:
            raise ValueError("The number of classes for a balanced replay buffer should be at least 2. "
                             "The number of classes that were defined are: {}".format(self.num_classes))

    def store(self, transition: Transition, lock: bool=True) -> None:
        """
        Store a new transition in the memory.
        :param transition: a transition to store
        :param lock: if true, will lock the readers writers lock. this can cause a deadlock if an inheriting class
                     locks and then calls store with lock = True
        :return: None
        """
        # Calling super.store() so that in case a memory backend is used, the memory backend can store this transition.
        super().store(transition)
        if lock:
            self.reader_writer_lock.lock_writing_and_reading()

        self._num_transitions += 1

        if self.state_key_with_the_class_index not in transition.state.keys():
            raise ValueError("The class index was not present in the state of the transition under the given key ({})"
                             .format(self.state_key_with_the_class_index))

        class_idx = transition.state[self.state_key_with_the_class_index]

        if class_idx >= self.num_classes:
            raise ValueError("The given class index is outside the defined number of classes for the replay buffer. "
                             "The given class was: {} and the number of classes defined is: {}"
                             .format(class_idx, self.num_classes))

        self.transitions[class_idx].append(transition)
        self.transitions_order.append(class_idx)
        self._enforce_max_length()

        if lock:
            self.reader_writer_lock.release_writing_and_reading()

    def sample(self, size: int) -> List[Transition]:
        """
        Sample a batch of transitions form the replay buffer. If the requested size is larger than the number
        of samples available in the replay buffer then the batch will return empty.
        :param size: the size of the batch to sample
        :return: a batch (list) of selected transitions from the replay buffer
        """
        self.reader_writer_lock.lock_writing()

        if size % self.num_classes != 0:
            raise ValueError("Sampling batches from a balanced replay buffer should be done only using batch sizes "
                             "which are a multiple of the number of classes. The number of classes defined is: {} "
                             "and the batch size requested is: {}".format(self.num_classes, size))

        batch_size_from_each_class = size // self.num_classes

        if self.allow_duplicates_in_batch_sampling:
            transitions_idx = [np.random.randint(len(class_transitions), size=batch_size_from_each_class)
                               for class_transitions in self.transitions]

        else:
            for class_idx, class_transitions in enumerate(self.transitions):
                if self.num_transitions() < batch_size_from_each_class:
                    raise ValueError("The replay buffer cannot be sampled since there are not enough transitions yet. "
                                     "There are currently {} transitions for class {}"
                                     .format(len(class_transitions), class_idx))

            transitions_idx = [np.random.choice(len(class_transitions), size=batch_size_from_each_class, replace=False)
                               for class_transitions in self.transitions]

        batch = []
        for class_idx, class_transitions_idx in enumerate(transitions_idx):
            batch += [self.transitions[class_idx][i] for i in class_transitions_idx]

        self.reader_writer_lock.release_writing()

        return batch

    def remove_transition(self, transition_index: int, lock: bool=True) -> None:
        raise ValueError("It is not possible to remove specific transitions with a balanced replay buffer")

    def get_transition(self, transition_index: int, lock: bool=True) -> Union[None, Transition]:
        raise ValueError("It is not possible to access specific transitions with a balanced replay buffer")

    def _enforce_max_length(self) -> None:
        """
        Make sure that the size of the replay buffer does not pass the maximum size allowed.
        If it passes the max size, the oldest transition in the replay buffer will be removed.
        This function does not use locks since it is only called internally
        :return: None
        """
        granularity, size = self.max_size
        if granularity == MemoryGranularity.Transitions:
            while size != 0 and self.num_transitions() > size:
                self._num_transitions -= 1
                del self.transitions[self.transitions_order[0]][0]
                del self.transitions_order[0]
        else:
            raise ValueError("The granularity of the replay buffer can only be set in terms of transitions")

    def clean(self, lock: bool=True) -> None:
        """
        Clean the memory by removing all the episodes
        :return: None
        """
        if lock:
            self.reader_writer_lock.lock_writing_and_reading()

        self.transitions = [[] for _ in range(self.num_classes)]
        self.transitions_order = []
        self._num_transitions = 0

        if lock:
            self.reader_writer_lock.release_writing_and_reading()
