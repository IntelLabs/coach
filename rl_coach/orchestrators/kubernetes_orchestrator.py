import os
import uuid
import json
import time
from typing import List
from rl_coach.orchestrators.deploy import Deploy, DeployParameters
from kubernetes import client, config
from rl_coach.memories.backend.memory import MemoryBackendParameters
from rl_coach.memories.backend.memory_impl import get_memory_backend


class RunTypeParameters():

    def __init__(self, image: str, command: list(), arguments: list() = None,
                 run_type: str = "trainer", checkpoint_dir: str = "/checkpoint",
                 num_replicas: int = 1, orchestration_params: dict=None):
        self.image = image
        self.command = command
        if not arguments:
            arguments = list()
        self.arguments = arguments
        self.run_type = run_type
        self.checkpoint_dir = checkpoint_dir
        self.num_replicas = num_replicas
        if not orchestration_params:
            orchestration_params = dict()
        self.orchestration_params = orchestration_params


class KubernetesParameters(DeployParameters):

    def __init__(self, run_type_params: List[RunTypeParameters], kubeconfig: str = None, namespace: str = "", nfs_server: str = None,
                 nfs_path: str = None, checkpoint_dir: str = '/checkpoint', memory_backend_parameters: MemoryBackendParameters = None):

        self.run_type_params = {}
        for run_type_param in run_type_params:
            self.run_type_params[run_type_param.run_type] = run_type_param
        self.kubeconfig = kubeconfig
        self.namespace = namespace
        self.nfs_server = nfs_server
        self.nfs_path = nfs_path
        self.checkpoint_dir = checkpoint_dir
        self.memory_backend_parameters = memory_backend_parameters


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

        if os.environ.get('http_proxy'):
            client.Configuration._default.proxy = os.environ.get('http_proxy')

        self.deploy_parameters.memory_backend_parameters.orchestrator_params = {'namespace': self.deploy_parameters.namespace}
        self.memory_backend = get_memory_backend(self.deploy_parameters.memory_backend_parameters)

    def setup(self) -> bool:

        self.memory_backend.deploy()
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

    def deploy_trainer(self) -> bool:

        trainer_params = self.deploy_parameters.run_type_params.get('trainer', None)
        if not trainer_params:
            return False

        trainer_params.command += ['--memory_backend_params', json.dumps(self.deploy_parameters.memory_backend_parameters.__dict__)]
        name = "{}-{}".format(trainer_params.run_type, uuid.uuid4())

        container = client.V1Container(
            name=name,
            image=trainer_params.image,
            command=trainer_params.command,
            args=trainer_params.arguments,
            image_pull_policy='Always',
            volume_mounts=[client.V1VolumeMount(
                name='nfs-pvc',
                mount_path=trainer_params.checkpoint_dir
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
            replicas=trainer_params.num_replicas,
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
            trainer_params.orchestration_params['deployment_name'] = name
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating deployment", e)
            return False

    def deploy_worker(self):

        worker_params = self.deploy_parameters.run_type_params.get('worker', None)
        if not worker_params:
            return False

        worker_params.command += ['--memory_backend_params', json.dumps(self.deploy_parameters.memory_backend_parameters.__dict__)]
        name = "{}-{}".format(worker_params.run_type, uuid.uuid4())

        container = client.V1Container(
            name=name,
            image=worker_params.image,
            command=worker_params.command,
            args=worker_params.arguments,
            image_pull_policy='Always',
            volume_mounts=[client.V1VolumeMount(
                name='nfs-pvc',
                mount_path=worker_params.checkpoint_dir
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
            ),
        )

        deployment_spec = client.V1DeploymentSpec(
            replicas=worker_params.num_replicas,
            template=template,
            selector=client.V1LabelSelector(
                match_labels={'app': name}
            )
        )
        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=name),
            spec=deployment_spec
        )

        api_client = client.AppsV1Api()
        try:
            api_client.create_namespaced_deployment(self.deploy_parameters.namespace, deployment)
            worker_params.orchestration_params['deployment_name'] = name
            return True
        except client.rest.ApiException as e:
            print("Got exception: %s\n while creating deployment", e)
            return False

    def worker_logs(self):
        pass

    def trainer_logs(self):
        trainer_params = self.deploy_parameters.run_type_params.get('trainer', None)
        if not trainer_params:
            return

        api_client = client.CoreV1Api()
        pod = None
        try:
            pods = api_client.list_namespaced_pod(self.deploy_parameters.namespace, label_selector='app={}'.format(
                trainer_params.orchestration_params['deployment_name']
            ))

            pod = pods.items[0]
        except client.rest.ApiException as e:
            print("Got exception: %s\n while reading pods", e)
            return

        if not pod:
            return

        self.tail_log(pod.metadata.name, api_client)

    def tail_log(self, pod_name, corev1_api):
        while True:
            time.sleep(10)
            # Try to tail the pod logs
            try:
                print(corev1_api.read_namespaced_pod_log(
                    pod_name, self.deploy_parameters.namespace, follow=True
                ), flush=True)
            except client.rest.ApiException as e:
                pass

            # This part will get executed if the pod is one of the following phases: not ready, failed or terminated.
            # Check if the pod has errored out, else just try again.
            # Get the pod
            try:
                pod = corev1_api.read_namespaced_pod(pod_name, self.deploy_parameters.namespace)
            except client.rest.ApiException as e:
                continue

            if not hasattr(pod, 'status') or not pod.status:
                continue
            if not hasattr(pod.status, 'container_statuses') or not pod.status.container_statuses:
                continue

            for container_status in pod.status.container_statuses:
                if container_status.state.waiting is not None:
                    if container_status.state.waiting.reason == 'Error' or \
                       container_status.state.waiting.reason == 'CrashLoopBackOff' or \
                       container_status.state.waiting.reason == 'ImagePullBackOff' or \
                       container_status.state.waiting.reason == 'ErrImagePull':
                        return
                if container_status.state.terminated is not None:
                    return

    def undeploy(self):
        trainer_params = self.deploy_parameters.run_type_params.get('trainer', None)
        api_client = client.AppsV1Api()
        delete_options = client.V1DeleteOptions()
        if trainer_params:
            try:
                api_client.delete_namespaced_deployment(trainer_params.orchestration_params['deployment_name'], self.deploy_parameters.namespace, delete_options)
            except client.rest.ApiException as e:
                print("Got exception: %s\n while deleting trainer", e)
        worker_params = self.deploy_parameters.run_type_params.get('worker', None)
        if worker_params:
            try:
                api_client.delete_namespaced_deployment(worker_params.orchestration_params['deployment_name'], self.deploy_parameters.namespace, delete_options)
            except client.rest.ApiException as e:
                print("Got exception: %s\n while deleting workers", e)
        self.memory_backend.undeploy()
