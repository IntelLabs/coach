
import redis
import pickle
import uuid
import time
from kubernetes import client

from rl_coach.memories.backend.memory import MemoryBackend, MemoryBackendParameters
from rl_coach.core_types import Transition, Episode, EnvironmentSteps, EnvironmentEpisodes


class RedisPubSubMemoryBackendParameters(MemoryBackendParameters):

    def __init__(self, redis_address: str="", redis_port: int=6379, channel: str="channel-{}".format(uuid.uuid4()),
                 orchestrator_params: dict=None, run_type='trainer', orchestrator_type: str = "kubernetes", deployed: str = False):
        self.redis_address = redis_address
        self.redis_port = redis_port
        self.channel = channel
        if not orchestrator_params:
            orchestrator_params = {}
        self.orchestrator_params = orchestrator_params
        self.run_type = run_type
        self.store_type = "redispubsub"
        self.orchestrator_type = orchestrator_type
        self.deployed = deployed


class RedisPubSubBackend(MemoryBackend):

    def __init__(self, params: RedisPubSubMemoryBackendParameters):
        self.params = params
        self.redis_connection = redis.Redis(self.params.redis_address, self.params.redis_port)
        self.redis_server_name = 'redis-server-{}'.format(uuid.uuid4())
        self.redis_service_name = 'redis-service-{}'.format(uuid.uuid4())

    def store(self, obj):
        self.redis_connection.publish(self.params.channel, pickle.dumps(obj))

    def deploy(self):
        if not self.params.deployed:
            if self.params.orchestrator_type == 'kubernetes':
                self.deploy_kubernetes()

        # Wait till subscribe to the channel is possible or else it will cause delays in the trainer.
        time.sleep(10)

    def deploy_kubernetes(self):

        if 'namespace' not in self.params.orchestrator_params:
            self.params.orchestrator_params['namespace'] = "default"

        container = client.V1Container(
            name=self.redis_server_name,
            image='redis:4-alpine',
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': self.redis_server_name}),
            spec=client.V1PodSpec(
                containers=[container]
            )
        )
        deployment_spec = client.V1DeploymentSpec(
            replicas=1,
            template=template,
            selector=client.V1LabelSelector(
                match_labels={'app': self.redis_server_name}
            )
        )

        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(name=self.redis_server_name, labels={'app': self.redis_server_name}),
            spec=deployment_spec
        )

        api_client = client.AppsV1Api()
        try:
            api_client.create_namespaced_deployment(self.params.orchestrator_params['namespace'], deployment)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating redis-server", e)
            return False

        core_v1_api = client.CoreV1Api()

        service = client.V1Service(
            api_version='v1',
            kind='Service',
            metadata=client.V1ObjectMeta(
                name=self.redis_service_name
            ),
            spec=client.V1ServiceSpec(
                selector={'app': self.redis_server_name},
                ports=[client.V1ServicePort(
                    protocol='TCP',
                    port=6379,
                    target_port=6379
                )]
            )
        )

        try:
            core_v1_api.create_namespaced_service(self.params.orchestrator_params['namespace'], service)
            self.params.redis_address = '{}.{}.svc'.format(
                self.redis_service_name, self.params.orchestrator_params['namespace']
            )
            self.params.redis_port = 6379
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating a service for redis-server", e)
            return False

    def undeploy(self):
        if self.params.deployed:
            return
        api_client = client.AppsV1Api()
        delete_options = client.V1DeleteOptions()
        try:
            api_client.delete_namespaced_deployment(self.redis_server_name, self.params.orchestrator_params['namespace'], delete_options)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while deleting redis-server", e)

        api_client = client.CoreV1Api()
        try:
            api_client.delete_namespaced_service(self.redis_service_name, self.params.orchestrator_params['namespace'], delete_options)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while deleting redis-server", e)

    def sample(self, size):
        pass

    def fetch(self, num_consecutive_playing_steps=None):
        return RedisSub(redis_address=self.params.redis_address, redis_port=self.params.redis_port, channel=self.params.channel).run(num_consecutive_playing_steps)

    def subscribe(self, agent):
        redis_sub = RedisSub(redis_address=self.params.redis_address, redis_port=self.params.redis_port, channel=self.params.channel)
        return redis_sub

    def get_endpoint(self):
        return {'redis_address': self.params.redis_address,
                'redis_port': self.params.redis_port}


class RedisSub(object):
    def __init__(self, redis_address: str = "localhost", redis_port: int=6379, channel: str = "PubsubChannel"):
        super().__init__()
        self.redis_connection = redis.Redis(redis_address, redis_port)
        self.pubsub = self.redis_connection.pubsub()
        self.subscriber = None
        self.channel = channel
        self.subscriber = self.pubsub.subscribe(self.channel)

    def run(self, num_consecutive_playing_steps):
        transitions = 0
        episodes = 0
        steps = 0
        for message in self.pubsub.listen():
            if message and 'data' in message:
                try:
                    obj = pickle.loads(message['data'])
                    if type(obj) == Transition:
                        transitions += 1
                        if obj.game_over:
                            episodes += 1
                        yield obj
                    elif type(obj) == Episode:
                        episodes += 1
                        transitions += len(obj.transitions)
                        yield from obj.transitions
                except Exception:
                    continue

            if type(num_consecutive_playing_steps) == EnvironmentSteps:
                steps = transitions
            if type(num_consecutive_playing_steps) == EnvironmentEpisodes:
                steps = episodes

            if steps >= num_consecutive_playing_steps.num_steps:
                break
