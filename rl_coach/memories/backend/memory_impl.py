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


from rl_coach.memories.backend.memory import MemoryBackendParameters
from rl_coach.memories.backend.redis import RedisPubSubBackend, RedisPubSubMemoryBackendParameters


def get_memory_backend(params: MemoryBackendParameters):

    backend = None
    if type(params) == RedisPubSubMemoryBackendParameters:
        backend = RedisPubSubBackend(params)

    return backend


def construct_memory_params(json: dict):

    if json['store_type'] == 'redispubsub':
        memory_params = RedisPubSubMemoryBackendParameters(
            json['redis_address'], json['redis_port'], channel=json.get('channel', ''), run_type=json['run_type']
        )
        return memory_params
