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


import uuid

from rl_coach.data_stores.data_store import DataStore, DataStoreParameters


class NFSDataStoreParameters(DataStoreParameters):
    def __init__(self, ds_params, deployed=False, server=None, path=None):
        super().__init__(ds_params.store_type, ds_params.orchestrator_type, ds_params.orchestrator_params)
        self.namespace = "default"
        if "namespace" in ds_params.orchestrator_params:
            self.namespace = ds_params.orchestrator_params["namespace"]
        self.name = None
        self.pvc_name = None
        self.pv_name = None
        self.svc_name = None
        self.server = None
        self.path = "/"
        self.deployed = deployed
        if deployed:
            self.server = server
            self.path = path


class NFSDataStore(DataStore):
    """
    An implementation of data store which uses NFS for storing policy checkpoints when using Coach in distributed mode.
    The policy checkpoints are written by the trainer and read by the rollout worker.
    """

    def __init__(self, params: NFSDataStoreParameters):
        """
        :param params: The parameters required to use the NFS data store.
        """
        self.params = params

    def deploy(self) -> bool:
        """
        Deploy the NFS server in an orchestrator if/when required.
        """
        if self.params.orchestrator_type == "kubernetes":
            if not self.params.deployed:
                if not self.deploy_k8s_nfs():
                    return False
            if not self.create_k8s_nfs_resources():
                return False

        return True

    def get_info(self):
        from kubernetes import client as k8sclient

        return k8sclient.V1PersistentVolumeClaimVolumeSource(
                claim_name=self.params.pvc_name
        )

    def undeploy(self) -> bool:
        """
        Undeploy the NFS server and resources from an orchestrator.
        """
        if self.params.orchestrator_type == "kubernetes":
            if not self.params.deployed:
                if not self.undeploy_k8s_nfs():
                    return False
            if not self.delete_k8s_nfs_resources():
                return False

        return True

    def save_to_store(self):
        pass

    def load_from_store(self):
        pass

    def deploy_k8s_nfs(self) -> bool:
        """
        Deploy the NFS server in the Kubernetes orchestrator.
        """
        from kubernetes import client as k8sclient

        name = "nfs-server-{}".format(uuid.uuid4())
        container = k8sclient.V1Container(
            name=name,
            image="k8s.gcr.io/volume-nfs:0.8",
            ports=[k8sclient.V1ContainerPort(
                    name="nfs",
                    container_port=2049,
                    protocol="TCP"
                   ),
                   k8sclient.V1ContainerPort(
                    name="rpcbind",
                    container_port=111
                   ),
                   k8sclient.V1ContainerPort(
                    name="mountd",
                    container_port=20048
                   ),
            ],
            volume_mounts=[k8sclient.V1VolumeMount(
                name='nfs-host-path',
                mount_path='/exports'
            )],
            security_context=k8sclient.V1SecurityContext(privileged=True)
        )
        template = k8sclient.V1PodTemplateSpec(
            metadata=k8sclient.V1ObjectMeta(labels={'app': name}),
            spec=k8sclient.V1PodSpec(
                containers=[container],
                volumes=[k8sclient.V1Volume(
                    name="nfs-host-path",
                    host_path=k8sclient.V1HostPathVolumeSource(path='/tmp/nfsexports-{}'.format(uuid.uuid4()))
                )]
            )
        )
        deployment_spec = k8sclient.V1DeploymentSpec(
            replicas=1,
            template=template,
            selector=k8sclient.V1LabelSelector(
                match_labels={'app': name}
            )
        )

        deployment = k8sclient.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=k8sclient.V1ObjectMeta(name=name, labels={'app': name}),
            spec=deployment_spec
        )

        k8s_apps_v1_api_client = k8sclient.AppsV1Api()
        try:
            k8s_apps_v1_api_client.create_namespaced_deployment(self.params.namespace, deployment)
            self.params.name = name
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating nfs-server", e)
            return False

        k8s_core_v1_api_client = k8sclient.CoreV1Api()

        svc_name = "nfs-service-{}".format(uuid.uuid4())
        service = k8sclient.V1Service(
            api_version='v1',
            kind='Service',
            metadata=k8sclient.V1ObjectMeta(
                name=svc_name
            ),
            spec=k8sclient.V1ServiceSpec(
                selector={'app': self.params.name},
                ports=[k8sclient.V1ServicePort(
                    protocol='TCP',
                    port=2049,
                    target_port=2049
                )]
            )
        )

        try:
            svc_response = k8s_core_v1_api_client.create_namespaced_service(self.params.namespace, service)
            self.params.svc_name = svc_name
            self.params.server = svc_response.spec.cluster_ip
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating a service for nfs-server", e)
            return False

        return True

    def create_k8s_nfs_resources(self) -> bool:
        """
        Create NFS resources such as PV and PVC in Kubernetes.
        """
        from kubernetes import client as k8sclient

        pv_name = "nfs-ckpt-pv-{}".format(uuid.uuid4())
        persistent_volume = k8sclient.V1PersistentVolume(
            api_version="v1",
            kind="PersistentVolume",
            metadata=k8sclient.V1ObjectMeta(
                name=pv_name,
                labels={'app': pv_name}
            ),
            spec=k8sclient.V1PersistentVolumeSpec(
                access_modes=["ReadWriteMany"],
                nfs=k8sclient.V1NFSVolumeSource(
                    path=self.params.path,
                    server=self.params.server
                ),
                capacity={'storage': '10Gi'},
                storage_class_name=""
            )
        )
        k8s_api_client = k8sclient.CoreV1Api()
        try:
            k8s_api_client.create_persistent_volume(persistent_volume)
            self.params.pv_name = pv_name
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating the NFS PV", e)
            return False

        pvc_name = "nfs-ckpt-pvc-{}".format(uuid.uuid4())
        persistent_volume_claim = k8sclient.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=k8sclient.V1ObjectMeta(
                name=pvc_name
            ),
            spec=k8sclient.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteMany"],
                resources=k8sclient.V1ResourceRequirements(
                    requests={'storage': '10Gi'}
                ),
                selector=k8sclient.V1LabelSelector(
                    match_labels={'app': self.params.pv_name}
                ),
                storage_class_name=""
            )
        )

        try:
            k8s_api_client.create_namespaced_persistent_volume_claim(self.params.namespace, persistent_volume_claim)
            self.params.pvc_name = pvc_name
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while creating the NFS PVC", e)
            return False

        return True

    def undeploy_k8s_nfs(self) -> bool:
        from kubernetes import client as k8sclient

        del_options = k8sclient.V1DeleteOptions()

        k8s_apps_v1_api_client = k8sclient.AppsV1Api()
        try:
            k8s_apps_v1_api_client.delete_namespaced_deployment(self.params.name, self.params.namespace, del_options)
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while deleting nfs-server", e)
            return False

        k8s_core_v1_api_client = k8sclient.CoreV1Api()
        try:
            k8s_core_v1_api_client.delete_namespaced_service(self.params.svc_name, self.params.namespace, del_options)
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while deleting the service for nfs-server", e)
            return False

        return True

    def delete_k8s_nfs_resources(self) -> bool:
        """
        Delete NFS resources such as PV and PVC from the Kubernetes orchestrator.
        """
        from kubernetes import client as k8sclient

        del_options = k8sclient.V1DeleteOptions()
        k8s_api_client = k8sclient.CoreV1Api()

        try:
            k8s_api_client.delete_persistent_volume(self.params.pv_name, del_options)
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while deleting NFS PV", e)
            return False

        try:
            k8s_api_client.delete_namespaced_persistent_volume_claim(self.params.pvc_name, self.params.namespace, del_options)
        except k8sclient.rest.ApiException as e:
            print("Got exception: %s\n while deleting NFS PVC", e)
            return False

        return True
