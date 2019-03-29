#
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
import ast
import math

import pandas as pd
from typing import List, Tuple, Union
import numpy as np
import random

from rl_coach.core_types import Transition, Episode
from rl_coach.logger import screen
from rl_coach.memories.memory import Memory, MemoryGranularity, MemoryParameters
from rl_coach.utils import ReaderWriterLock, ProgressBar
from rl_coach.core_types import CsvDataset


class EpisodicExperienceReplayParameters(MemoryParameters):
    def __init__(self):
        super().__init__()
        self.max_size = (MemoryGranularity.Transitions, 1000000)
        self.n_step = -1
        self.train_to_eval_ratio = 1  # for OPE we'll want a value < 1

    @property
    def path(self):
        return 'rl_coach.memories.episodic.episodic_experience_replay:EpisodicExperienceReplay'


class EpisodicExperienceReplay(Memory):
    """
    A replay buffer that stores episodes of transitions. The additional structure allows performing various
    calculations of total return and other values that depend on the sequential behavior of the transitions
    in the episode.
    """

    def __init__(self, max_size: Tuple[MemoryGranularity, int] = (MemoryGranularity.Transitions, 1000000), n_step=-1,
                 train_to_eval_ratio: int = 1):
        """
        :param max_size: the maximum number of transitions or episodes to hold in the memory
        """
        super().__init__(max_size)
        self.n_step = n_step
        self._buffer = [Episode(n_step=self.n_step)]  # list of episodes
        self.transitions = []
        self._length = 1  # the episodic replay buffer starts with a single empty episode
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0
        self.reader_writer_lock = ReaderWriterLock()
        self.last_training_set_episode_id = None  # used in batch-rl
        self.last_training_set_transition_id = None  # used in batch-rl
        self.train_to_eval_ratio = train_to_eval_ratio  # used in batch-rl

    def length(self, lock: bool = False) -> int:
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

    def get_last_training_set_episode_id(self):
        return self.last_training_set_episode_id

    def sample(self, size: int, is_consecutive_transitions=False) -> List[Transition]:
        """
        Sample a batch of transitions from the replay buffer. If the requested size is larger than the number
        of samples available in the replay buffer then the batch will return empty.
        :param size: the size of the batch to sample
        :param is_consecutive_transitions: if set True, samples a batch of consecutive transitions.
        :return: a batch (list) of selected transitions from the replay buffer
        """
        self.reader_writer_lock.lock_writing()

        if self.num_complete_episodes() >= 1:
            if is_consecutive_transitions:
                episode_idx = np.random.randint(0, self.num_complete_episodes())
                if self._buffer[episode_idx].length() <= size:
                    batch = self._buffer[episode_idx].transitions
                else:
                    transition_idx = np.random.randint(size, self._buffer[episode_idx].length())
                    batch = self._buffer[episode_idx].transitions[transition_idx - size:transition_idx]
            else:
                transitions_idx = np.random.randint(self.num_transitions_in_complete_episodes(), size=size)
                batch = [self.transitions[i] for i in transitions_idx]

        else:
            raise ValueError("The episodic replay buffer cannot be sampled since there are no complete episodes yet. "
                             "There is currently 1 episodes with {} transitions".format(self._buffer[0].length()))

        self.reader_writer_lock.release_writing()

        return batch

    def get_episode_for_transition(self, transition: Transition) -> (int, Episode):
        """
        Get the episode from which that transition came from.
        :param transition: The transition to lookup the episode for
        :return: (Episode number, the episode) or (-1, None) if could not find a matching episode.
        """

        for i, episode in enumerate(self._buffer):
            if transition in episode.transitions:
                return i, episode
        return -1, None

    def shuffle_episodes(self):
        """
        Shuffle all the episodes in the replay buffer
        :return:
        """
        random.shuffle(self._buffer)
        self.transitions = [t for e in self._buffer for t in e.transitions]

    def get_shuffled_data_generator(self, size: int) -> List[Transition]:
        """
        Get an generator for iterating through the shuffled replay buffer, for processing the data in epochs.
        If the requested size is larger than the number of samples available in the replay buffer then the batch will
        return empty. The last returned batch may be smaller than the size requested, to accommodate for all the
        transitions in the replay buffer.

        :param size: the size of the batch to return
        :return: a batch (list) of selected transitions from the replay buffer
        """
        self.reader_writer_lock.lock_writing()
        if self.last_training_set_transition_id is None:
            if self.train_to_eval_ratio < 0 or self.train_to_eval_ratio >= 1:
                raise ValueError('train_to_eval_ratio should be in the (0, 1] range.')

            transition = self.transitions[round(self.train_to_eval_ratio * self.num_transitions_in_complete_episodes())]
            episode_num, episode = self.get_episode_for_transition(transition)
            self.last_training_set_episode_id = episode_num
            self.last_training_set_transition_id = \
                len([t for e in self.get_all_complete_episodes_from_to(0, self.last_training_set_episode_id + 1) for t in e])

        shuffled_transition_indices = list(range(self.last_training_set_transition_id))
        random.shuffle(shuffled_transition_indices)

        # The last batch drawn will usually be < batch_size (=the size variable)
        for i in range(math.ceil(len(shuffled_transition_indices) / size)):
            sample_data = [self.transitions[j] for j in shuffled_transition_indices[i * size: (i + 1) * size]]
            self.reader_writer_lock.release_writing()

            yield sample_data

    def get_all_complete_episodes_transitions(self) -> List[Transition]:
        """
        Get all the transitions from all the complete episodes in the buffer
        :return: a list of transitions
        """
        return self.transitions[:self.num_transitions_in_complete_episodes()]

    def get_all_complete_episodes(self) -> List[Episode]:
        """
        Get all the transitions from all the complete episodes in the buffer
        :return: a list of transitions
        """
        return self.get_all_complete_episodes_from_to(0, self.num_complete_episodes())

    def get_all_complete_episodes_from_to(self, start_episode_id, end_episode_id) -> List[Episode]:
        """
        Get all the transitions from all the complete episodes in the buffer matching the given episode range
        :return: a list of transitions
        """
        return self._buffer[start_episode_id:end_episode_id]

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
        episode.update_transitions_rewards_and_bootstrap_data()

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
        self._buffer.append(Episode(n_step=self.n_step))

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

        # Calling super.store() so that in case a memory backend is used, the memory backend can store this transition.
        super().store(transition)

        self.reader_writer_lock.lock_writing_and_reading()

        if len(self._buffer) == 0:
            self._buffer.append(Episode(n_step=self.n_step))
        last_episode = self._buffer[-1]
        last_episode.insert(transition)
        self.transitions.append(transition)
        self._num_transitions += 1
        if transition.game_over:
            self.close_last_episode(False)

        self._enforce_max_length()

        self.reader_writer_lock.release_writing_and_reading()

    def store_episode(self, episode: Episode, lock: bool = True) -> None:
        """
        Store a new episode in the memory.
        :param episode: the new episode to store
        :return: None
        """
        # Calling super.store() so that in case a memory backend is used, the memory backend can store this episode.
        super().store_episode(episode)

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

    def get_episode(self, episode_index: int, lock: bool = True) -> Union[None, Episode]:
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
    def get(self, episode_index: int, lock: bool = True) -> Union[None, Episode]:
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

    def clean(self) -> None:
        """
        Clean the memory by removing all the episodes
        :return: None
        """
        self.reader_writer_lock.lock_writing_and_reading()

        self.transitions = []
        self._buffer = [Episode(n_step=self.n_step)]
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

    def load_csv(self, csv_dataset: CsvDataset) -> None:
        """
        Restore the replay buffer contents from a csv file.
        The csv file is assumed to include a list of transitions.
        :param csv_dataset: A construct which holds the dataset parameters
        """
        df = pd.read_csv(csv_dataset.filepath)
        if len(df) > self.max_size[1]:
            screen.warning("Warning! The number of transitions to load into the replay buffer ({}) is "
                           "bigger than the max size of the replay buffer ({}). The excessive transitions will "
                           "not be stored.".format(len(df), self.max_size[1]))

        episode_ids = df['episode_id'].unique()
        progress_bar = ProgressBar(len(episode_ids))
        state_columns = [col for col in df.columns if col.startswith('state_feature')]

        for e_id in episode_ids:
            progress_bar.update(e_id)
            df_episode_transitions = df[df['episode_id'] == e_id]
            episode = Episode()
            for (_, current_transition), (_, next_transition) in zip(df_episode_transitions[:-1].iterrows(),
                                                                     df_episode_transitions[1:].iterrows()):
                state = np.array([current_transition[col] for col in state_columns])
                next_state = np.array([next_transition[col] for col in state_columns])

                episode.insert(
                    Transition(state={'observation': state},
                               action=current_transition['action'], reward=current_transition['reward'],
                               next_state={'observation': next_state}, game_over=False,
                               info={'all_action_probabilities':
                                         ast.literal_eval(current_transition['all_action_probabilities'])}))

            # Set the last transition to end the episode
            if csv_dataset.is_episodic:
                episode.get_last_transition().game_over = True

            self.store_episode(episode)

        # close the progress bar
        progress_bar.update(len(episode_ids))
        progress_bar.close()

        self.shuffle_episodes()
