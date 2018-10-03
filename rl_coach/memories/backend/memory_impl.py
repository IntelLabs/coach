
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
