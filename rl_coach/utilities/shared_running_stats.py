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
from abc import ABC, abstractmethod
import threading
import pickle
import redis
import numpy as np


class SharedRunningStatsSubscribe(threading.Thread):
    def __init__(self, shared_running_stats):
        super().__init__()
        self.shared_running_stats = shared_running_stats
        self.redis_address = self.shared_running_stats.pubsub.params.redis_address
        self.redis_port = self.shared_running_stats.pubsub.params.redis_port
        self.redis_connection = redis.Redis(self.redis_address, self.redis_port)
        self.pubsub = self.redis_connection.pubsub()
        self.channel = self.shared_running_stats.channel
        self.pubsub.subscribe(self.channel)

    def run(self):
        for message in self.pubsub.listen():
            try:
                obj = pickle.loads(message['data'])
                self.shared_running_stats.push_val(obj)
            except Exception:
                continue


class SharedRunningStats(ABC):
    def __init__(self, name="", pubsub_params=None):
        self.name = name
        self.pubsub = None
        if pubsub_params:
            self.channel = "channel-srs-{}".format(self.name)
            from rl_coach.memories.backend.memory_impl import get_memory_backend
            self.pubsub = get_memory_backend(pubsub_params)
            subscribe_thread = SharedRunningStatsSubscribe(self)
            subscribe_thread.daemon = True
            subscribe_thread.start()

    @abstractmethod
    def set_params(self, shape=[1], clip_values=None):
        pass

    def push(self, x):
        if self.pubsub:
            self.pubsub.redis_connection.publish(self.channel, pickle.dumps(x))
            return

        self.push_val(x)

    @abstractmethod
    def push_val(self, x):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass

    @property
    @abstractmethod
    def var(self):
        pass

    @property
    @abstractmethod
    def std(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def normalize(self, batch):
        pass

    @abstractmethod
    def set_session(self, sess):
        pass

    @abstractmethod
    def save_state_to_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: int):
        pass

    @abstractmethod
    def restore_state_from_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: str):
        pass

    def get_latest_checkpoint(self, checkpoint_dir: str) -> str:
        latest_checkpoint_id = -1
        latest_checkpoint = ''
        # get all checkpoint files
        for fname in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, fname)
            if os.path.isdir(path) or fname.split('.')[-1] != 'srs':
                continue
            checkpoint_id = int(fname.split('_')[0])
            if checkpoint_id > latest_checkpoint_id:
                latest_checkpoint = fname
                latest_checkpoint_id = checkpoint_id

        return latest_checkpoint


class NumpySharedRunningStats(SharedRunningStats):
    def __init__(self, name, epsilon=1e-2, pubsub_params=None):
        super().__init__(name=name, pubsub_params=pubsub_params)
        self._count = epsilon
        self.epsilon = epsilon

    def set_params(self, shape=[1], clip_values=None):
        self._shape = shape
        self._mean = np.zeros(shape)
        self._std = np.sqrt(self.epsilon) * np.ones(shape)
        self._sum = np.zeros(shape)
        self._sum_squares = self.epsilon * np.ones(shape)
        self.clip_values = clip_values

    def push_val(self, samples: np.ndarray):
        assert len(samples.shape) >= 2  # we should always have a batch dimension
        assert samples.shape[1:] == self._mean.shape, 'RunningStats input shape mismatch'
        self._sum += samples.sum(axis=0).ravel()
        self._sum_squares += np.square(samples).sum(axis=0).ravel()
        self._count += np.shape(samples)[0]
        self._mean = self._sum / self._count
        self._std = np.sqrt(np.maximum(
            (self._sum_squares - self._count * np.square(self._mean)) / np.maximum(self._count - 1, 1),
            self.epsilon))

    @property
    def n(self):
        return self._count

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._std ** 2

    @property
    def std(self):
        return self._std

    @property
    def shape(self):
        return self._mean.shape

    def normalize(self, batch):
        batch = (batch - self.mean) / (self.std + 1e-15)
        return np.clip(batch, *self.clip_values)

    def set_session(self, sess):
        # no session for the numpy implementation
        pass

    def save_state_to_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: int):
        dict_to_save = {'_mean': self._mean,
                        '_std': self._std,
                        '_count': self._count,
                        '_sum': self._sum,
                        '_sum_squares': self._sum_squares}

        with open(os.path.join(checkpoint_dir, str(checkpoint_prefix) + '.srs'), 'wb') as f:
            pickle.dump(dict_to_save, f, pickle.HIGHEST_PROTOCOL)

    def restore_state_from_checkpoint(self, checkpoint_dir: str, checkpoint_prefix: str):
        latest_checkpoint_filename = self.get_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint_filename == '':
            raise ValueError("Could not find NumpySharedRunningStats checkpoint file. ")

        with open(os.path.join(checkpoint_dir, str(latest_checkpoint_filename)), 'rb') as f:
            saved_dict = pickle.load(f)
            self.__dict__.update(saved_dict)
