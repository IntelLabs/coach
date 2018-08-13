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

from typing import Tuple

import tensorflow as tf


def create_cluster_spec(parameters_server: str, workers: str) -> tf.train.ClusterSpec:
    """
    Creates a ClusterSpec object representing the cluster.
    :param parameters_server: comma-separated list of hostname:port pairs to which the parameter servers are assigned
    :param workers: comma-separated list of hostname:port pairs to which the workers are assigned
    :return: a ClusterSpec object representing the cluster
    """
    # extract the parameter servers and workers from the given strings
    ps_hosts = parameters_server.split(",")
    worker_hosts = workers.split(",")

    # Create a cluster spec from the parameter server and worker hosts
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    return cluster_spec


def create_and_start_parameters_server(cluster_spec: tf.train.ClusterSpec, config: tf.ConfigProto=None) -> None:
    """
    Create and start a parameter server
    :param cluster_spec: the ClusterSpec object representing the cluster
    :param config: the tensorflow config to use
    :return: None
    """
    # create a server object for the parameter server
    server = tf.train.Server(cluster_spec, job_name="ps", task_index=0, config=config)

    # wait for the server to finish
    server.join()


def create_worker_server_and_device(cluster_spec: tf.train.ClusterSpec, task_index: int,
                                    use_cpu: bool=True, config: tf.ConfigProto=None) -> Tuple[str, tf.device]:
    """
    Creates a worker server and a device setter used to assign the workers operations to
    :param cluster_spec: a ClusterSpec object representing the cluster
    :param task_index: the index of the worker task
    :param use_cpu: if use_cpu=True, all the agent operations will be assigned to a CPU instead of a GPU
    :param config: the tensorflow config to use
    :return: the target string for the tf.Session and the worker device setter object
    """
    # Create and start a worker
    server = tf.train.Server(cluster_spec, job_name="worker", task_index=task_index, config=config)

    # Assign ops to the local worker
    worker_device = "/job:worker/task:{}".format(task_index)
    if use_cpu:
        worker_device += "/cpu:0"
    else:
        worker_device += "/device:GPU:0"
    device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster_spec)

    return server.target, device


def create_monitored_session(target: tf.train.Server, task_index: int,
                             checkpoint_dir: str, save_checkpoint_secs: int, config: tf.ConfigProto=None) -> tf.Session:
    """
    Create a monitored session for the worker
    :param target: the target string for the tf.Session
    :param task_index: the task index of the worker
    :param checkpoint_dir: a directory path where the checkpoints will be stored
    :param save_checkpoint_secs: number of seconds between checkpoints storing
    :param config: the tensorflow configuration (optional)
    :return: the session to use for the run
    """
    # we chose the first task to be the chief
    is_chief = task_index == 0

    # Create the monitored session
    sess = tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=is_chief,
        hooks=[],
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=save_checkpoint_secs,
        config=config
    )

    return sess

