
import uuid
from rl_coach.orchestrators.deploy import Deploy, DeployParameters
from kubernetes import client, config


class KubernetesParameters(DeployParameters):

    def __init__(self, name: str, image: str, command: list(), arguments: list() = list(),  synchronized: bool = False,
                 num_workers: int = 1, kubeconfig: str = None, namespace: str = None, redis_ip: str = None,
                 redis_port: int = None, redis_db: int = 0, nfs_server: str = None, nfs_path: str = None,
                 checkpoint_dir: str = '/checkpoint'):
        self.image = image
        self.synchronized = synchronized
        self.command = command
        self.arguments = arguments
        self.kubeconfig = kubeconfig
        self.num_workers = num_workers
        self.namespace = namespace
        self.redis_ip = redis_ip
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.nfs_server = nfs_server
        self.nfs_path = nfs_path
        self.checkpoint_dir = checkpoint_dir
        self.name = name


class Kubernetes(Deploy):

    def __init__(self, deploy_parameters: KubernetesParameters):
        super().__init__(deploy_parameters)
        self.deploy_parameters = deploy_parameters
        if self.deploy_parameters.kubeconfig:
            config.load_kube_config()
        else:
            config.load_incluster_config()

        if not self.deploy_parameters.namespace:
            _, current_context = config.list_kube_config_contexts()
            self.deploy_parameters.namespace = current_context['context']['namespace']
        self.nfs_pvc_name = 'nfs-checkpoint-pvc'

    def setup(self) -> bool:

        if not self.deploy_parameters.redis_ip:
            # Need to spin up a redis service and a deployment.
            if not self.deploy_redis():
                print("Failed to setup redis")
                return False

        if not self.create_nfs_resources():
            return False

        return True

    def create_nfs_resources(self):
        persistent_volume = client.V1PersistentVolume(
            api_version="v1",
            kind="PersistentVolume",
            metadata=client.V1ObjectMeta(
                name='nfs-checkpoint-pv',
                labels={'app': 'nfs-checkpoint-pv'}
            ),
            spec=client.V1PersistentVolumeSpec(
                access_modes=["ReadWriteMany"],
                nfs=client.V1NFSVolumeSource(
                    path=self.deploy_parameters.nfs_path,
                    server=self.deploy_parameters.nfs_server
                ),
                capacity={'storage': '10Gi'},
                storage_class_name=""
            )
        )
        api_client = client.CoreV1Api()
        try:
            api_client.create_persistent_volume(persistent_volume)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating the NFS PV", e)
            return False

        persistent_volume_claim = client.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=client.V1ObjectMeta(
                name="nfs-checkpoint-pvc"
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteMany"],
                resources=client.V1ResourceRequirements(
                    requests={'storage': '10Gi'}
                ),
                selector=client.V1LabelSelector(
                    match_labels={'app': 'nfs-checkpoint-pv'}
                ),
                storage_class_name=""
            )
        )

        try:
            api_client.create_namespaced_persistent_volume_claim(self.deploy_parameters.namespace, persistent_volume_claim)
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating the NFS PVC", e)
            return False
        return True

    def deploy_redis(self) -> bool:
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
            api_client.create_namespaced_deployment(self.deploy_parameters.namespace, deployment)
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
            core_v1_api.create_namespaced_service(self.deploy_parameters.namespace, service)
            self.deploy_parameters.redis_ip = 'redis-service.{}.svc'.format(self.deploy_parameters.namespace)
            self.deploy_parameters.redis_port = 6379
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating a service for redis-server", e)
            return False

    def deploy(self) -> bool:

        self.deploy_parameters.command += ['--redis_ip', self.deploy_parameters.redis_ip, '--redis_port', '{}'.format(self.deploy_parameters.redis_port)]

        if self.deploy_parameters.synchronized:
            return self.create_k8s_deployment()
        else:
            return self.create_k8s_job()

    def create_k8s_deployment(self) -> bool:
        name = "{}-{}".format(self.deploy_parameters.name, uuid.uuid4())

        container = client.V1Container(
            name=name,
            image=self.deploy_parameters.image,
            command=self.deploy_parameters.command,
            args=self.deploy_parameters.arguments,
            image_pull_policy='Always',
            volume_mounts=[client.V1VolumeMount(
                name='nfs-pvc',
                mount_path=self.deploy_parameters.checkpoint_dir
            )]
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': name}),
            spec=client.V1PodSpec(
                containers=[container],
                volumes=[client.V1Volume(
                    name="nfs-pvc",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.nfs_pvc_name
                    )
                )]
            ),
        )
        deployment_spec = client.V1DeploymentSpec(
            replicas=self.deploy_parameters.num_workers,
            template=template,
            selector=client.V1LabelSelector(
                match_labels={'app': name}
            )
        )

        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(name=name),
            spec=deployment_spec
        )

        api_client = client.AppsV1Api()
        try:
            api_client.create_namespaced_deployment(self.deploy_parameters.namespace, deployment)
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating deployment", e)
            return False

    def create_k8s_job(self):
        name = "{}-{}".format(self.deploy_parameters.name, uuid.uuid4())

        container = client.V1Container(
            name=name,
            image=self.deploy_parameters.image,
            command=self.deploy_parameters.command,
            args=self.deploy_parameters.arguments,
            image_pull_policy='Always',
            volume_mounts=[client.V1VolumeMount(
                name='nfs-pvc',
                mount_path=self.deploy_parameters.checkpoint_dir
            )]
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': name}),
            spec=client.V1PodSpec(
                containers=[container],
                volumes=[client.V1Volume(
                    name="nfs-pvc",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.nfs_pvc_name
                    )
                )],
                restart_policy='Never'
            ),
        )

        job_spec = client.V1JobSpec(
            parallelism=self.deploy_parameters.num_workers,
            template=template,
            completions=2147483647
        )

        job = client.V1Job(
            api_version='batch/v1',
            kind='Job',
            metadata=client.V1ObjectMeta(name=name),
            spec=job_spec
        )

        api_client = client.BatchV1Api()
        try:
            api_client.create_namespaced_job(self.deploy_parameters.namespace, job)
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating deployment", e)
            return False
