
import argparse
import pytest
import time
from kubernetes import client, config


class EKSHandler():

    def __init__(self, cluster, build_num, test_name, test_command, image, cpu, memory, working_dir):
        self.cluster = cluster
        self.build_num = build_num
        self.test_name = test_name
        self.test_command = test_command
        self.image = image
        self.cpu = cpu
        self.memory = memory
        config.load_kube_config()
        self.namespace = '{}-{}'.format(test_name, build_num)
        self.corev1_api = client.CoreV1Api()
        self.create_namespace()
        self.working_dir = working_dir

    def create_namespace(self):
        namespace = client.V1Namespace(
            api_version='v1',
            kind="Namespace",
            metadata=client.V1ObjectMeta(name=self.namespace)
        )

        try:
            self.corev1_api.create_namespace(namespace)
        except client.rest.ApiException as e:
            raise RuntimeError("Failed to create namesapce. Got exception: {}".format(e))

    def deploy(self):
        container = client.V1Container(
            name=self.test_name,
            image=self.image,
            command=['/bin/bash', '-c'],
            args=[self.test_command],
            image_pull_policy='Always',
            working_dir=self.working_dir,
            stdin=True,
            tty=True
        )
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy='Never'
        )
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=self.test_name),
            spec=pod_spec
        )

        try:
            self.corev1_api.create_namespaced_pod(self.namespace, pod)
        except client.rest.ApiException as e:
            print("Got exception: {} while creating a pod".format(e))
            return 1

        return 0

    def print_logs(self):
        while True:
            time.sleep(10)
            # Try to tail the pod logs
            try:
                for line in self.corev1_api.read_namespaced_pod_log(
                    self.test_name, self.namespace, follow=True,
                    _preload_content=False
                ):
                    print(line.decode('utf-8'), flush=True, end='')

            except client.rest.ApiException as e:
                pass

            try:
                pod = self.corev1_api.read_namespaced_pod(self.test_name, self.namespace)
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

    def get_return_status(self):
        # This part will get executed if the pod is one of the following phases: not ready, failed or terminated.
        # Check if the pod has errored out, else just try again.
        # Get the pod
        try:
            pod = self.corev1_api.read_namespaced_pod(self.test_name, self.namespace)
        except client.rest.ApiException as e:
            return 1

        if not hasattr(pod, 'status') or not pod.status:
            return 0
        if not hasattr(pod.status, 'container_statuses') or not pod.status.container_statuses:
            return 0

        for container_status in pod.status.container_statuses:
            if container_status.state.waiting is not None:
                if container_status.state.waiting.reason == 'Error' or \
                   container_status.state.waiting.reason == 'CrashLoopBackOff' or \
                   container_status.state.waiting.reason == 'ImagePullBackOff' or \
                   container_status.state.waiting.reason == 'ErrImagePull':
                    return 1
            if container_status.state.terminated is not None:
                return container_status.state.terminated.exit_code

    def cleanup(self):

        # Delete pod
        try:
            self.corev1_api.delete_namespaced_pod(self.test_name, self.namespace, client.V1DeleteOptions())
        except client.rest.ApiException as e:
            print("Got exception while deleting pod: {}".format(e))

        # Delete namespace
        try:
            self.corev1_api.delete_namespace(self.namespace, client.V1DeleteOptions())
        except client.rest.ApiException as e:
            print("Got exception while deleting namespace: {}".format(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--cluster', help="(string) Name of the cluster", type=str, required=True
    )
    parser.add_argument(
        '-bn', '--build-num', help="(int) CI Build number", type=int, required=True
    )
    parser.add_argument(
        '-tn', '--test-name', help="(string) Name of the test", type=str, required=True
    )
    parser.add_argument(
        '-tc', '--test-command', help="(string) command to execute", type=str, required=True
    )
    parser.add_argument(
        '-i', '--image', help="(string) Container image", type=str, required=True
    )
    parser.add_argument(
        '-cpu', help="(string) Units of cpu to use", type=str, required=True
    )
    parser.add_argument(
        '-mem', help="(string) The amount in megabytes", type=str, required=True
    )
    parser.add_argument(
        '--working-dir', help="(string) The working dir in the container", type=str, required=False,
        default='/root/src/docker'
    )
    args = parser.parse_args()

    obj = EKSHandler(
        args.cluster, args.build_num, args.test_name, args.test_command,
        args.image, args.cpu, args.mem, args.working_dir
    )

    if obj.deploy() != 0:
        obj.cleanup()
        pytest.fail("Failed to deploy")

    obj.print_logs()

    if obj.get_return_status() != 0:
        obj.cleanup()
        pytest.fail("Failed to run tests")

    obj.cleanup()
