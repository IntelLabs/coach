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
import uuid
import json
import time
import sys
from enum import Enum
from typing import List
from configparser import ConfigParser, Error
from multiprocessing import Process

from rl_coach.base_parameters import RunType
from rl_coach.orchestrators.deploy import Deploy, DeployParameters
from kubernetes import client as k8sclient, config as k8sconfig
from rl_coach.memories.backend.memory import MemoryBackendParameters
from rl_coach.memories.backend.memory_impl import get_memory_backend
from rl_coach.data_stores.data_store import DataStoreParameters
from rl_coach.data_stores.data_store_impl import get_data_store


class RunTypeParameters():

    def __init__(self, image: str, command: list(), arguments: list() = None,
                 run_type: str = str(RunType.TRAINER), checkpoint_dir: str = "/checkpoint",
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

    def __init__(self, run_type_params: List[RunTypeParameters], kubeconfig: str = None, namespace: str = None,
                 nfs_server: str = None, nfs_path: str = None, checkpoint_dir: str = '/checkpoint',
                 memory_backend_parameters: MemoryBackendParameters = None, data_store_params: DataStoreParameters = None):

        self.run_type_params = {}
        for run_type_param in run_type_params:
            self.run_type_params[run_type_param.run_type] = run_type_param
        self.kubeconfig = kubeconfig
        self.namespace = namespace
        self.nfs_server = nfs_server
        self.nfs_path = nfs_path
        self.checkpoint_dir = checkpoint_dir
        self.memory_backend_parameters = memory_backend_parameters
        self.data_store_params = data_store_params


class Kubernetes(Deploy):
    """
    An orchestrator implmentation which uses Kubernetes to deploy the components such as training and rollout workers
    and Redis Pub/Sub in Coach when used in the distributed mode.
    """

    def __init__(self, params: KubernetesParameters):
        """
        :param params: The Kubernetes parameters which are used for deploying the components in Coach. These parameters
        include namespace and kubeconfig.
        """

        super().__init__(params)
        self.params = params
        if self.params.kubeconfig:
            k8sconfig.load_kube_config()
        else:
            k8sconfig.load_incluster_config()

        if not self.params.namespace:
            _, current_context = k8sconfig.list_kube_config_contexts()
            self.params.namespace = current_context['context']['namespace']

        if os.environ.get('http_proxy'):
            k8sclient.Configuration._default.proxy = os.environ.get('http_proxy')

        self.params.memory_backend_parameters.orchestrator_params = {'namespace': self.params.namespace}
        self.memory_backend = get_memory_backend(self.params.memory_backend_parameters)

        self.params.data_store_params.orchestrator_params = {'namespace': self.params.namespace}
        self.params.data_store_params.namespace = self.params.namespace
        self.data_store = get_data_store(self.params.data_store_params)

        if self.params.data_store_params.store_type == "s3":
            self.s3_access_key = None
            self.s3_secret_key = None
            if self.params.data_store_params.creds_file:
                s3config = ConfigParser()
                s3config.read(self.params.data_store_params.creds_file)
                try:
                    self.s3_access_key = s3config.get('default', 'aws_access_key_id')
                    self.s3_secret_key = s3config.get('default', 'aws_secret_access_key')
                except Error as e:
                    print("Error when reading S3 credentials file: %s", e)
            else:
                self.s3_access_key = os.environ.get('ACCESS_KEY_ID')
                self.s3_secret_key = os.environ.get('SECRET_ACCESS_KEY')

    def setup(self) -> bool:
        """
        Deploys the memory backend and data stores if required.
        """

        self.memory_backend.deploy()
        if not self.data_store.deploy():
            return False
        if self.params.data_store_params.store_type == "nfs":
            self.nfs_pvc = self.data_store.get_info()
        return True

    def deploy_trainer(self) -> bool:
        """
        Deploys the training worker in Kubernetes.
        """

        trainer_params = self.params.run_type_params.get(str(RunType.TRAINER), None)
        if not trainer_params:
            return False

        trainer_params.command += ['--memory_backend_params', json.dumps(self.params.memory_backend_parameters.__dict__)]
        trainer_params.command += ['--data_store_params', json.dumps(self.params.data_store_params.__dict__)]

        name = "{}-{}".format(trainer_params.run_type, uuid.uuid4())

        if self.params.data_store_params.store_type == "nfs":
            container = k8sclient.V1Container(
                name=name,
                image=trainer_params.image,
                command=trainer_params.command,
                args=trainer_params.arguments,
                image_pull_policy='Always',
                volume_mounts=[k8sclient.V1VolumeMount(
                    name='nfs-pvc',
                    mount_path=trainer_params.checkpoint_dir
                )],
                stdin=True,
                tty=True
            )
            template = k8sclient.V1PodTemplateSpec(
                metadata=k8sclient.V1ObjectMeta(labels={'app': name}),
                spec=k8sclient.V1PodSpec(
                    containers=[container],
                    volumes=[k8sclient.V1Volume(
                        name="nfs-pvc",
                        persistent_volume_claim=self.nfs_pvc
                    )],
                    restart_policy='OnFailure'
                ),
            )
        else:
            container = k8sclient.V1Container(
                name=name,
                image=trainer_params.image,
                command=trainer_params.command,
                args=trainer_params.arguments,
                image_pull_policy='Always',
                env=[k8sclient.V1EnvVar("ACCESS_KEY_ID", self.s3_access_key),
                     k8sclient.V1EnvVar("SECRET_ACCESS_KEY", self.s3_secret_key)],
                stdin=True,
                tty=True
            )
            template = k8sclient.V1PodTemplateSpec(
                metadata=k8sclient.V1ObjectMeta(labels={'app': name}),
                spec=k8sclient.V1PodSpec(
                    containers=[container],
                    restart_policy='OnFailure'
                ),
            )

        job_spec = k8sclient.V1JobSpec(
            completions=1,
            template=template
        )

        job = k8sclient.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8sclient.V1ObjectMeta(name=name),
            spec=job_spec
        )

        api_client = k8sclient.BatchV1Api()
        try:
            api_client.create_namespaced_job(self.params.namespace, job)
            trainer_params.orchestration_params['job_name'] = name
            return True
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating job", e)
            return False

    def deploy_worker(self):
        """
        Deploys the rollout worker(s) in Kubernetes.
        """

        worker_params = self.params.run_type_params.get(str(RunType.ROLLOUT_WORKER), None)
        if not worker_params:
            return False

        worker_params.command += ['--memory_backend_params', json.dumps(self.params.memory_backend_parameters.__dict__)]
        worker_params.command += ['--data_store_params', json.dumps(self.params.data_store_params.__dict__)]
        worker_params.command += ['--num_workers', '{}'.format(worker_params.num_replicas)]

        name = "{}-{}".format(worker_params.run_type, uuid.uuid4())

        if self.params.data_store_params.store_type == "nfs":
            container = k8sclient.V1Container(
                name=name,
                image=worker_params.image,
                command=worker_params.command,
                args=worker_params.arguments,
                image_pull_policy='Always',
                volume_mounts=[k8sclient.V1VolumeMount(
                    name='nfs-pvc',
                    mount_path=worker_params.checkpoint_dir
                )],
                stdin=True,
                tty=True
            )
            template = k8sclient.V1PodTemplateSpec(
                metadata=k8sclient.V1ObjectMeta(labels={'app': name}),
                spec=k8sclient.V1PodSpec(
                    containers=[container],
                    volumes=[k8sclient.V1Volume(
                        name="nfs-pvc",
                        persistent_volume_claim=self.nfs_pvc
                    )],
                    restart_policy='OnFailure'
                ),
            )
        else:
            container = k8sclient.V1Container(
                name=name,
                image=worker_params.image,
                command=worker_params.command,
                args=worker_params.arguments,
                image_pull_policy='Always',
                env=[k8sclient.V1EnvVar("ACCESS_KEY_ID", self.s3_access_key),
                     k8sclient.V1EnvVar("SECRET_ACCESS_KEY", self.s3_secret_key)],
                stdin=True,
                tty=True
            )
            template = k8sclient.V1PodTemplateSpec(
                metadata=k8sclient.V1ObjectMeta(labels={'app': name}),
                spec=k8sclient.V1PodSpec(
                    containers=[container],
                    restart_policy='OnFailure'
                )
            )

        job_spec = k8sclient.V1JobSpec(
            completions=worker_params.num_replicas,
            parallelism=worker_params.num_replicas,
            template=template
        )

        job = k8sclient.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8sclient.V1ObjectMeta(name=name),
            spec=job_spec
        )

        api_client = k8sclient.BatchV1Api()
        try:
            api_client.create_namespaced_job(self.params.namespace, job)
            worker_params.orchestration_params['job_name'] = name
            return True
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating Job", e)
            return False

    def worker_logs(self, path='./logs'):
        """
        :param path: Path to store the worker logs.
        """
        worker_params = self.params.run_type_params.get(str(RunType.ROLLOUT_WORKER), None)
        if not worker_params:
            return

        api_client = k8sclient.CoreV1Api()
        pods = None
        try:
            pods = api_client.list_namespaced_pod(self.params.namespace, label_selector='app={}'.format(
                worker_params.orchestration_params['job_name']
            ))

            # pod = pods.items[0]
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while reading pods", e)
            return

        if not pods or len(pods.items) == 0:
            return

        for pod in pods.items:
            Process(target=self._tail_log_file, args=(pod.metadata.name, api_client, self.params.namespace, path)).start()

    def _tail_log_file(self, pod_name, api_client, namespace, path):
        if not os.path.exists(path):
            os.mkdir(path)

        sys.stdout = open(os.path.join(path, pod_name), 'w')
        self.tail_log(pod_name, api_client)

    def trainer_logs(self):
        """
        Get the logs from trainer.
        """
        trainer_params = self.params.run_type_params.get(str(RunType.TRAINER), None)
        if not trainer_params:
            return

        api_client = k8sclient.CoreV1Api()
        pod = None
        try:
            pods = api_client.list_namespaced_pod(self.params.namespace, label_selector='app={}'.format(
                trainer_params.orchestration_params['job_name']
            ))

            pod = pods.items[0]
        except k8sclient.rest.ApiException as e:
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
                for line in corev1_api.read_namespaced_pod_log(
                            pod_name, self.params.namespace, follow=True,
                            _preload_content=False
                        ):
                    print(line.decode('utf-8'), flush=True, end='')
            except k8sclient.rest.ApiException as e:
                pass

            # This part will get executed if the pod is one of the following phases: not ready, failed or terminated.
            # Check if the pod has errored out, else just try again.
            # Get the pod
            try:
                pod = corev1_api.read_namespaced_pod(pod_name, self.params.namespace)
            except k8sclient.rest.ApiException as e:
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
        """
        Undeploy all the components, such as trainer and rollout worker(s), Redis pub/sub and data store, when required.
        """

        trainer_params = self.params.run_type_params.get(str(RunType.TRAINER), None)
        api_client = k8sclient.BatchV1Api()
        delete_options = k8sclient.V1DeleteOptions(
            propagation_policy="Foreground"
        )

        if trainer_params:
            try:
                api_client.delete_namespaced_job(trainer_params.orchestration_params['job_name'], self.params.namespace, delete_options)
            except k8sclient.rest.ApiException as e:
                print("Got exception: %s\n while deleting trainer", e)
        worker_params = self.params.run_type_params.get(str(RunType.ROLLOUT_WORKER), None)
        if worker_params:
            try:
                api_client.delete_namespaced_job(worker_params.orchestration_params['job_name'], self.params.namespace, delete_options)
            except k8sclient.rest.ApiException as e:
                print("Got exception: %s\n while deleting workers", e)
        self.memory_backend.undeploy()
        self.data_store.undeploy()
