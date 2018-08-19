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

from typing import List, Tuple, Union, Dict, Any

import numpy as np

from rl_coach.core_types import Transition, Episode
from rl_coach.memories.memory import Memory, MemoryGranularity, MemoryParameters
from rl_coach.utils import ReaderWriterLock


class EpisodicExperienceReplayParameters(MemoryParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 1000000)

    @property
    def path(self):
        return 'rl_coach.memories.episodic.episodic_experience_replay:EpisodicExperienceReplay'


class EpisodicExperienceReplay(Memory):
    """
    A replay buffer that stores episodes of transitions. The additional structure allows performing various
    calculations of total return and other values that depend on the sequential behavior of the transitions
    in the episode.
    """
    def __init__(self, max_size: Tuple[MemoryGranularity, int]):
        """
        :param max_size: the maximum number of transitions or episodes to hold in the memory
        """
        super().__init__(max_size)

        self._buffer = [Episode()]  # list of episodes
        self.transitions = []
        self._length = 1  # the episodic replay buffer starts with a single empty episode
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0

        self.reader_writer_lock = ReaderWriterLock()

    def length(self, lock: bool=False) -> int:
        """
        Get the number of episodes in the ER (even if they are not complete)
        """
        length = self._length
        if self._length is not 0 and self._buffer[-1].is_empty():
            length = self._length - 1

        return length

    def num_complete_episodes(self):
        """ Get the number of complete episodes in ER """
        length = self._length - 1

        return length

    def num_transitions(self):
        return self._num_transitions

    def num_transitions_in_complete_episodes(self):
        return self._num_transitions_in_complete_episodes

    def sample(self, size: int) -> List[Transition]:
        """
        Sample a batch of transitions form the replay buffer. If the requested size is larger than the number
        of samples available in the replay buffer then the batch will return empty.
        :param size: the size of the batch to sample
        :return: a batch (list) of selected transitions from the replay buffer
        """
        self.reader_writer_lock.lock_writing()

        if self.num_complete_episodes() >= 1:
            transitions_idx = np.random.randint(self.num_transitions_in_complete_episodes(), size=size)
            batch = [self.transitions[i] for i in transitions_idx]

        else:
            raise ValueError("The episodic replay buffer cannot be sampled since there are no complete episodes yet. "
                             "There is currently 1 episodes with {} transitions".format(self._buffer[0].length()))

        self.reader_writer_lock.release_writing()

        return batch

    def _enforce_max_length(self) -> None:
        """
        Make sure that the size of the replay buffer does not pass the maximum size allowed.
        If it passes the max size, the oldest episode in the replay buffer will be removed.
        :return: None
        """
        granularity, size = self.max_size
        if granularity == MemoryGranularity.Transitions:
            while size != 0 and self.num_transitions() > size:
                self._remove_episode(0)
        elif granularity == MemoryGranularity.Episodes:
            while self.length() > size:
                self._remove_episode(0)

    def _update_episode(self, episode: Episode) -> None:
        episode.update_returns()

    def verify_last_episode_is_closed(self) -> None:
        """
        Verify that there is no open episodes in the replay buffer
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        last_episode = self.get(-1, False)
        if last_episode and last_episode.length() > 0:
            self.close_last_episode(lock=False)

        self.reader_writer_lock.release_writing_and_reading()

    def close_last_episode(self, lock=True) -> None:
        """
        Close the last episode in the replay buffer and open a new one
        :return: None
        """
        if lock:
            self.reader_writer_lock.lock_writing_and_reading()

        last_episode = self._buffer[-1]

        self._num_transitions_in_complete_episodes += last_episode.length()
        self._length += 1

        # create a new Episode for the next transitions to be placed into
        self._buffer.append(Episode())

        # if update episode adds to the buffer, a new Episode needs to be ready first
        # it would be better if this were less state full
        self._update_episode(last_episode)

        self._enforce_max_length()

        if lock:
            self.reader_writer_lock.release_writing_and_reading()

    def store(self, transition: Transition) -> None:
        """
        Store a new transition in the memory. If the transition game_over flag is on, this closes the episode and
        creates a new empty episode.
        Warning! using the episodic memory by storing individual transitions instead of episodes will use the default
        Episode class parameters in order to create new episodes.
        :param transition: a transition to store
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        if len(self._buffer) == 0:
            self._buffer.append(Episode())
        last_episode = self._buffer[-1]
        last_episode.insert(transition)
        self.transitions.append(transition)
        self._num_transitions += 1
        if transition.game_over:
            self.close_last_episode(False)

        self._enforce_max_length()

        self.reader_writer_lock.release_writing_and_reading()

    def store_episode(self, episode: Episode, lock: bool=True) -> None:
        """
        Store a new episode in the memory.
        :param episode: the new episode to store
        :return: None
        """
        if lock:
            self.reader_writer_lock.lock_writing_and_reading()

        if self._buffer[-1].length() == 0:
            self._buffer[-1] = episode
        else:
            self._buffer.append(episode)
        self.transitions.extend(episode.transitions)
        self._num_transitions += episode.length()
        self.close_last_episode(False)

        if lock:
            self.reader_writer_lock.release_writing_and_reading()

    def get_episode(self, episode_index: int, lock: bool=True) -> Union[None, Episode]:
        """
        Returns the episode in the given index. If the episode does not exist, returns None instead.
        :param episode_index: the index of the episode to return
        :return: the corresponding episode
        """
        if lock:
            self.reader_writer_lock.lock_writing()

        if self.length() == 0 or episode_index >= self.length():
            episode = None
        else:
            episode = self._buffer[episode_index]

        if lock:
            self.reader_writer_lock.release_writing()
        return episode

    def _remove_episode(self, episode_index: int) -> None:
        """
        Remove the episode in the given index (even if it is not complete yet)
        :param episode_index: the index of the episode to remove
        :return: None
        """
        if len(self._buffer) > episode_index:
            episode_length = self._buffer[episode_index].length()
            self._length -= 1
            self._num_transitions -= episode_length
            self._num_transitions_in_complete_episodes -= episode_length
            del self.transitions[:episode_length]
            del self._buffer[episode_index]

    def remove_episode(self, episode_index: int) -> None:
        """
        Remove the episode in the given index (even if it is not complete yet)
        :param episode_index: the index of the episode to remove
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        self._remove_episode(episode_index)

        self.reader_writer_lock.release_writing_and_reading()

    # for API compatibility
    def get(self, episode_index: int, lock: bool=True) -> Union[None, Episode]:
        """
        Returns the episode in the given index. If the episode does not exist, returns None instead.
        :param episode_index: the index of the episode to return
        :return: the corresponding episode
        """
        return self.get_episode(episode_index, lock)

    def get_last_complete_episode(self) -> Union[None, Episode]:
        """
        Returns the last complete episode in the memory or None if there are no complete episodes
        :return: None or the last complete episode
        """
        self.reader_writer_lock.lock_writing()

        last_complete_episode_index = self.num_complete_episodes() - 1
        episode = None
        if last_complete_episode_index >= 0:
            episode = self.get(last_complete_episode_index)

        self.reader_writer_lock.release_writing()

        return episode

    # for API compatibility
    def remove(self, episode_index: int):
        """
        Remove the episode in the given index (even if it is not complete yet)
        :param episode_index: the index of the episode to remove
        :return: None
        """
        self.remove_episode(episode_index)

    def update_last_transition_info(self, info: Dict[str, Any]) -> None:
        """
        Update the info of the last transition stored in the memory
        :param info: the new info to append to the existing info
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        episode = self._buffer[-1]
        if episode.length() == 0:
            if len(self._buffer) < 2:
                return
            episode = self._buffer[-2]
        episode.transitions[-1].info.update(info)

        self.reader_writer_lock.release_writing_and_reading()

    def clean(self) -> None:
        """
        Clean the memory by removing all the episodes
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        self.transitions = []
        self._buffer = [Episode()]
        self._length = 1
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0

        self.reader_writer_lock.release_writing_and_reading()

    def mean_reward(self) -> np.ndarray:
        """
        Get the mean reward in the replay buffer
        :return: the mean reward
        """
        self.reader_writer_lock.lock_writing()

        mean = np.mean([transition.reward for transition in self.transitions])

        self.reader_writer_lock.release_writing()
        return mean
