
import redis
import pickle
import uuid
import threading
from kubernetes import client

from rl_coach.memories.backend.memory import MemoryBackend, MemoryBackendParameters
from rl_coach.memories.memory import Memory
from rl_coach.core_types import Transition, Episode


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

    def store(self, obj):
        self.redis_connection.publish(self.params.channel, pickle.dumps(obj))

    def deploy(self):
        if not self.params.deployed:
            if self.params.orchestrator_type == 'kubernetes':
                self.deploy_kubernetes()
        self.params.deployed = True

    def deploy_kubernetes(self):

        if 'namespace' not in self.params.orchestrator_params:
            self.params.orchestrator_params['namespace'] = "default"

        container = client.V1Container(
            name="redis-server",
            image='redis:4-alpine',
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': 'redis-server'}),
            spec=client.V1PodSpec(
                containers=[container]
            )
        )
        deployment_spec = client.V1DeploymentSpec(
            replicas=1,
            template=template,
            selector=client.V1LabelSelector(
                match_labels={'app': 'redis-server'}
            )
        )

        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(name='redis-server', labels={'app': 'redis-server'}),
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
                name='redis-service'
            ),
            spec=client.V1ServiceSpec(
                selector={'app': 'redis-server'},
                ports=[client.V1ServicePort(
                    protocol='TCP',
                    port=6379,
                    target_port=6379
                )]
            )
        )

        try:
            core_v1_api.create_namespaced_service(self.params.orchestrator_params['namespace'], service)
            self.params.redis_address = 'redis-service.{}.svc'.format(self.params.orchestrator_params['namespace'])
            self.params.redis_port = 6379
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating a service for redis-server", e)
            return False

    def undeploy(self):
        if not self.params.deployed:
            return
        api_client = client.AppsV1Api()
        delete_options = client.V1DeleteOptions()
        try:
            api_client.delete_namespaced_deployment('redis-server', self.params.orchestrator_params['namespace'], delete_options)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while deleting redis-server", e)

        api_client = client.CoreV1Api()
        try:
            api_client.delete_namespaced_service('redis-service', self.params.orchestrator_params['namespace'], delete_options)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while deleting redis-server", e)

        self.params.deployed = False

    def sample(self, size):
        pass

    def subscribe(self, memory):
        redis_sub = RedisSub(memory, redis_address=self.params.redis_address, redis_port=self.params.redis_port, channel=self.params.channel)
        redis_sub.daemon = True
        redis_sub.start()

    def get_endpoint(self):
        return {'redis_address': self.params.redis_address,
                'redis_port': self.params.redis_port}


class RedisSub(threading.Thread):

    def __init__(self, memory: Memory, redis_address: str = "localhost", redis_port: int=6379, channel: str = "PubsubChannel"):
        super().__init__()
        self.redis_connection = redis.Redis(redis_address, redis_port)
        self.pubsub = self.redis_connection.pubsub()
        self.subscriber = None
        self.memory = memory
        self.channel = channel
        self.subscriber = self.pubsub.subscribe(self.channel)

    def run(self):
        for message in self.pubsub.listen():
            if message and 'data' in message:
                try:
                    obj = pickle.loads(message['data'])
                    if type(obj) == Transition:
                        self.memory.store(obj)
                    elif type(obj) == Episode:
                        self.memory.store_episode(obj)
                except Exception:
                    continue
