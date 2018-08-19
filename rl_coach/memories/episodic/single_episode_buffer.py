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

from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplay
from rl_coach.memories.memory import MemoryGranularity, MemoryParameters


class SingleEpisodeBufferParameters(MemoryParameters):
    def __init__(self):
        super().__init__()
        del self.max_size

    @property
    def path(self):
        return 'rl_coach.memories.episodic.single_episode_buffer:SingleEpisodeBuffer'


class SingleEpisodeBuffer(EpisodicExperienceReplay):
    def __init__(self):
        super().__init__((MemoryGranularity.Episodes, 1))
