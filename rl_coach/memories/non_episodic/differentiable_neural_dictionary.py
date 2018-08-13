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

import os
import pickle

import numpy as np
from annoy import AnnoyIndex


class AnnoyDictionary(object):
    def __init__(self, dict_size, key_width, new_value_shift_coefficient=0.1, batch_size=100, key_error_threshold=0.01,
                 num_neighbors=50, override_existing_keys=True, rebuild_on_every_update=False):
        self.rebuild_on_every_update = rebuild_on_every_update
        self.max_size = dict_size
        self.curr_size = 0
        self.new_value_shift_coefficient = new_value_shift_coefficient
        self.num_neighbors = num_neighbors
        self.override_existing_keys = override_existing_keys

        self.index = AnnoyIndex(key_width, metric='euclidean')
        self.index.set_seed(1)

        self.embeddings = np.zeros((dict_size, key_width))
        self.values = np.zeros(dict_size)
        self.additional_data = [None] * dict_size

        self.lru_timestamps = np.zeros(dict_size)
        self.current_timestamp = 0.0

        # keys that are in this distance will be considered as the same key
        self.key_error_threshold = key_error_threshold

        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size
        self.key_dimension = key_width
        self.value_dimension = 1
        self._reset_buffer()

        self.built_capacity = 0

    def add(self, keys, values, additional_data=None):
        if not additional_data:
            additional_data = [None] * len(keys)

        # Adds new embeddings and values to the dictionary
        indices = []
        indices_to_remove = []
        for i in range(keys.shape[0]):
            index = self._lookup_key_index(keys[i])
            if index and self.override_existing_keys:
                # update existing value
                self.values[index] += self.new_value_shift_coefficient * (values[i] - self.values[index])
                self.additional_data[index[0][0]] = additional_data[i]
                self.lru_timestamps[index] = self.current_timestamp
                indices_to_remove.append(i)
            else:
                # add new
                if self.curr_size >= self.max_size:
                    # find the LRU entry
                    index = np.argmin(self.lru_timestamps)
                else:
                    index = self.curr_size
                    self.curr_size += 1
                self.lru_timestamps[index] = self.current_timestamp
                indices.append(index)

        for i in reversed(indices_to_remove):
            keys = np.delete(keys, i, 0)
            values = np.delete(values, i, 0)
            del additional_data[i]

        self.buffered_keys = np.vstack((self.buffered_keys, keys))
        self.buffered_values = np.vstack((self.buffered_values, values))
        self.buffered_indices = self.buffered_indices + indices
        self.buffered_additional_data = self.buffered_additional_data + additional_data

        if len(self.buffered_indices) >= self.min_update_size:
            self.min_update_size = max(self.initial_update_size, int(self.curr_size * 0.02))
            self._rebuild_index()
        elif self.rebuild_on_every_update:
            self._rebuild_index()

        self.current_timestamp += 1

    # Returns the stored embeddings and values of the closest embeddings
    def query(self, keys, k):
        if not self.has_enough_entries(k):
            # this will only happen when the DND is not yet populated with enough entries, which is only during heatup
            # these values won't be used and therefore they are meaningless
            return [0.0], [0.0], [0], [None]

        _, indices = self._get_k_nearest_neighbors_indices(keys, k)

        embeddings = []
        values = []
        additional_data = []
        for ind in indices:
            self.lru_timestamps[ind] = self.current_timestamp
            embeddings.append(self.embeddings[ind])
            values.append(self.values[ind])
            curr_additional_data = []
            for sub_ind in ind:
                curr_additional_data.append(self.additional_data[sub_ind])
            additional_data.append(curr_additional_data)

        self.current_timestamp += 1

        return embeddings, values, indices, additional_data

    def has_enough_entries(self, k):
        return self.curr_size > k and (self.built_capacity > k)

    def sample_embeddings(self, num_embeddings):
        return self.embeddings[np.random.choice(self.curr_size, num_embeddings)]

    def _get_k_nearest_neighbors_indices(self, keys, k):
        distances = []
        indices = []
        for key in keys:
            index, distance = self.index.get_nns_by_vector(key, k, include_distances=True)
            distances.append(distance)
            indices.append(index)
        return distances, indices

    def _rebuild_index(self):
        self.index.unbuild()
        self.embeddings[self.buffered_indices] = self.buffered_keys
        self.values[self.buffered_indices] = np.squeeze(self.buffered_values)
        for i, data in zip(self.buffered_indices, self.buffered_additional_data):
            self.additional_data[i] = data
        for idx, key in zip(self.buffered_indices, self.buffered_keys):
            self.index.add_item(idx, key)

        self._reset_buffer()

        self.index.build(self.num_neighbors)
        self.built_capacity = self.curr_size

    def _reset_buffer(self):
        self.buffered_keys = np.zeros((0, self.key_dimension))
        self.buffered_values = np.zeros((0, self.value_dimension))
        self.buffered_indices = []
        self.buffered_additional_data = []

    def _lookup_key_index(self, key):
        distance, index = self._get_k_nearest_neighbors_indices([key], 1)
        if distance != [[]] and distance[0][0] <= self.key_error_threshold:
            return index
        return None


class QDND(object):
    def __init__(self, dict_size, key_width, num_actions, new_value_shift_coefficient=0.1, key_error_threshold=0.01,
                 learning_rate=0.01, num_neighbors=50, return_additional_data=False, override_existing_keys=False,
                 rebuild_on_every_update=False):
        self.dict_size = dict_size
        self.key_width = key_width
        self.num_actions = num_actions
        self.new_value_shift_coefficient = new_value_shift_coefficient
        self.key_error_threshold = key_error_threshold
        self.learning_rate = learning_rate
        self.num_neighbors = num_neighbors
        self.return_additional_data = return_additional_data
        self.override_existing_keys = override_existing_keys
        self.dicts = []

        # create a dict for each action
        for a in range(num_actions):
            new_dict = AnnoyDictionary(dict_size, key_width, new_value_shift_coefficient,
                                       key_error_threshold=key_error_threshold, num_neighbors=num_neighbors,
                                       override_existing_keys=override_existing_keys,
                                       rebuild_on_every_update=rebuild_on_every_update)
            self.dicts.append(new_dict)

    def add(self, embeddings, actions, values, additional_data=None):
        # add a new set of embeddings and values to each of the underlining dictionaries
        embeddings = np.array(embeddings)
        actions = np.array(actions)
        values = np.array(values)
        for a in range(self.num_actions):
            idx = np.where(actions == a)
            curr_action_embeddings = embeddings[idx]
            curr_action_values = np.expand_dims(values[idx], -1)
            if additional_data:
                curr_additional_data = []
                for i in idx[0]:
                    curr_additional_data.append(additional_data[i])
            else:
                curr_additional_data = None

            self.dicts[a].add(curr_action_embeddings, curr_action_values, curr_additional_data)
        return True

    def query(self, embeddings, action, k):
        # query for nearest neighbors to the given embeddings
        dnd_embeddings = []
        dnd_values = []
        dnd_indices = []
        dnd_additional_data = []
        for i in range(len(embeddings)):
            embedding, value, indices, additional_data = self.dicts[action].query([embeddings[i]], k)
            dnd_embeddings.append(embedding[0])
            dnd_values.append(value[0])
            dnd_indices.append(indices[0])
            dnd_additional_data.append(additional_data[0])

        if self.return_additional_data:
            return dnd_embeddings, dnd_values, dnd_indices, dnd_additional_data
        else:
            return dnd_embeddings, dnd_values, dnd_indices

    def has_enough_entries(self, k):
        # check if each of the action dictionaries has at least k entries
        for a in range(self.num_actions):
            if not self.dicts[a].has_enough_entries(k):
                return False
        return True

    def update_keys_and_values(self, actions, key_gradients, value_gradients, indices):
        # Update DND keys and values
        for batch_action, batch_keys, batch_values, batch_indices in zip(actions, key_gradients, value_gradients, indices):
            # Update keys (embeddings) and values in DND
            for i, index in enumerate(batch_indices):
                self.dicts[batch_action].embeddings[index, :] -= self.learning_rate * batch_keys[i, :]
                self.dicts[batch_action].values[index] -= self.learning_rate * batch_values[i]

    def sample_embeddings(self, num_embeddings):
        num_actions = len(self.dicts)
        embeddings = []
        num_embeddings_per_action = int(num_embeddings/num_actions)
        for action in range(num_actions):
            embeddings.append(self.dicts[action].sample_embeddings(num_embeddings_per_action))
        embeddings = np.vstack(embeddings)

        # the numbers did not divide nicely, let's just randomly sample some more embeddings
        if num_embeddings_per_action * num_actions < num_embeddings:
            action = np.random.randint(0, num_actions)
            extra_embeddings = self.dicts[action].sample_embeddings(num_embeddings -
                                                                   num_embeddings_per_action * num_actions)
            embeddings = np.vstack([embeddings, extra_embeddings])
        return embeddings

    def clean(self):
        # create a new dict for each action
        self.dicts = []
        for a in range(self.num_actions):
            new_dict = AnnoyDictionary(self.dict_size, self.key_width, self.new_value_shift_coefficient,
                                       key_error_threshold=self.key_error_threshold, num_neighbors=self.num_neighbors)
            self.dicts.append(new_dict)


def load_dnd(model_dir):
    max_id = 0

    for f in [s for s in os.listdir(model_dir) if s.endswith('.dnd')]:
        if int(f.split('.')[0]) > max_id:
            max_id = int(f.split('.')[0])

    model_path = str(max_id) + '.dnd'
    with open(os.path.join(model_dir, model_path), 'rb') as f:
        DND = pickle.load(f)

        for a in range(DND.num_actions):
            DND.dicts[a].index = AnnoyIndex(512, metric='euclidean')
            DND.dicts[a].index.set_seed(1)

            for idx, key in zip(range(DND.dicts[a].curr_size), DND.dicts[a].embeddings[:DND.dicts[a].curr_size]):
                DND.dicts[a].index.add_item(idx, key)

            DND.dicts[a].index.build(50)

    return DND