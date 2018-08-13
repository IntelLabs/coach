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

from enum import Enum
from typing import Tuple

from rl_coach.base_parameters import Parameters


class MemoryGranularity(Enum):
    Transitions = 0
    Episodes = 1


class MemoryParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.max_size = None
        self.shared_memory = False
        self.load_memory_from_file_path = None


    @property
    def path(self):
        return 'rl_coach.memories.memory:Memory'


class Memory(object):
    def __init__(self, max_size: Tuple[MemoryGranularity, int]):
        """
        :param max_size: the maximum number of objects to hold in the memory
        """
        self.max_size = max_size
        self._length = 0

    def store(self, obj):
        raise NotImplementedError("")

    def get(self, index):
        raise NotImplementedError("")

    def remove(self, index):
        raise NotImplementedError("")

    def length(self):
        raise NotImplementedError("")

    def sample(self, size):
        raise NotImplementedError("")

    def clean(self):
        raise NotImplementedError("")


