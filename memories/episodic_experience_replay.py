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

from memories.memory import *
import threading
from typing import Union


class EpisodicExperienceReplay(Memory):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        """
        Memory.__init__(self, tuning_parameters)
        self.tp = tuning_parameters
        self.max_size_in_episodes = tuning_parameters.agent.num_episodes_in_experience_replay
        self.max_size_in_transitions = tuning_parameters.agent.num_transitions_in_experience_replay
        self.discount = tuning_parameters.agent.discount
        self.buffer = [Episode()]  # list of episodes
        self.transitions = []
        self._length = 1
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0
        self.return_is_bootstrapped = tuning_parameters.agent.bootstrap_total_return_from_old_policy

    def length(self):
        """ Get the number of episodes in the ER (even if they are not complete) """
        if self._length is not 0 and self.buffer[-1].is_empty():
            return self._length - 1
        return self._length

    def num_complete_episodes(self):
        """ Get the number of complete episodes in ER """
        return self._length - 1

    def num_transitions(self):
        return self._num_transitions

    def num_transitions_in_complete_episodes(self):
        return self._num_transitions_in_complete_episodes

    def sample_episode(self):
        episode_idx = np.random.randint(self.num_complete_episodes())
        return self.buffer[episode_idx]

    def sample_n_episodes(self, n):
        num_n_episodes = (self.num_complete_episodes()) / n
        assert num_n_episodes > 0, \
            'Tried sampling {} episodes when only {} completed episodes are available in the memory' \
                .format(n, self.num_complete_episodes())
        start_episode_idx = np.random.randint(num_n_episodes) * n
        return self.buffer[start_episode_idx:(start_episode_idx + n)]

    def sample_last_n_episodes(self, n):
        num_n_episodes = (self.num_complete_episodes()) / n
        assert num_n_episodes > 0, \
            'Tried sampling {} episodes when only {} completed episodes are available in the memory' \
                .format(n, self.num_complete_episodes())
        start_episode_idx = -n
        return self.buffer[start_episode_idx:(start_episode_idx + n)]

    def sample(self, size):
        assert self.num_transitions_in_complete_episodes() > size, \
            'There are not enough transitions in the replay buffer. ' \
            'Available transitions: {}. Requested transitions: {}.'\
                .format(self.num_transitions_in_complete_episodes(), size)
        batch = []
        transitions_idx = np.random.randint(self.num_transitions_in_complete_episodes(), size=size)
        for i in transitions_idx:
            batch.append(self.transitions[i])

        return batch

    def enforce_length(self):
        # clean up if necessary
        if self.max_size_in_transitions is not None:
            while self.max_size_in_transitions != 0 and self.num_transitions() > self.max_size_in_transitions:
                self.remove_episode(0)
        else:
            while self.length() > self.max_size_in_episodes:
                self.remove_episode(0)

    def store(self, transition):
        if len(self.buffer) == 0:
            self.buffer.append(Episode())
        last_episode = self.buffer[-1]
        last_episode.insert(transition)
        self.transitions.append(transition)
        self._num_transitions += 1
        if transition.game_over:
            self._num_transitions_in_complete_episodes += last_episode.length()
            self._length += 1
            self.buffer[-1].update_returns(self.discount,
                                           is_bootstrapped=self.tp.agent.bootstrap_total_return_from_old_policy,
                                           n_step_return=self.tp.agent.n_step)
            self.buffer[-1].update_measurements_targets(self.tp.agent.num_predicted_steps_ahead)
            # self.buffer[-1].update_actions_probabilities() # used for off-policy policy optimization
            self.buffer.append(Episode())

        self.enforce_length()

    def insert_full_episode(self, episode):
        # Do not add a new episode if the last one is not closed yet
        if self.buffer[-1].get_last_transition().done != True:
            return False

        episode.update_returns(self.discount)
        episode.update_measurements_targets(self.tp.agent.num_predicted_steps_ahead)
        self.buffer.append(episode)
        self.transitions += episode.transitions
        self._length += 1
        self._num_transitions += episode.length()

        self.enforce_length()

        return True

    def get_episode(self, episode_index):
        if self.length() == 0:
            return None
        episode = self.buffer[episode_index]
        return episode

    def remove_episode(self, episode_index):
        if len(self.buffer) > episode_index:
            episode_length = self.buffer[episode_index].length()
            self._length -= 1
            self._num_transitions -= episode_length
            self._num_transitions_in_complete_episodes -= episode_length
            del self.transitions[:episode_length]
            del self.buffer[episode_index]

    # for API compatibility
    def get(self, index):
        return self.get_episode(index)

    def get_last_complete_episode(self) -> Union[None, Episode]:
        """
        Returns the last complete episode in the memory or None if there are no complete episodes
        :return: None or the last complete episode
        """
        last_complete_episode_index = self.num_complete_episodes()-1
        if last_complete_episode_index >= 0:
            return self.get(last_complete_episode_index)
        else:
            return None

    def update_last_transition_info(self, info):
        episode = self.buffer[-1]
        if episode.length() == 0:
            if len(self.buffer) < 2:
                return
            episode = self.buffer[-2]
        for key, val in info.items():
            episode.transitions[-1].info[key] = val

    def clean(self):
        self.transitions = []
        self.buffer = [Episode()]
        self._length = 1
        self._num_transitions = 0
        self._num_transitions_in_complete_episodes = 0
