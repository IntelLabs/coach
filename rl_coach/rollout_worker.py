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

"""
this rollout worker:

- restores a model from disk
- evaluates a predefined number of episodes
- contributes them to a distributed memory
- exits
"""



import time
import os

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType
from rl_coach.checkpoint import CheckpointStateFile, CheckpointStateReader
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.core_types import RunPhase


def wait_for(wait_func, data_store=None, timeout=10):
    """
    block until wait_func is true
    """
    for i in range(timeout):
        if data_store:
            data_store.load_from_store()

        if wait_func():
            return
        time.sleep(10)

    # one last time
    if wait_func():
        return

    raise ValueError((
        'Waited {timeout} seconds, but condition timed out'
    ).format(
        timeout=timeout,
    ))


def wait_for_trainer_ready(checkpoint_dir, data_store=None, timeout=10):
    """
    Block until trainer is ready
    """

    def wait():
        return os.path.exists(os.path.join(checkpoint_dir, SyncFiles.TRAINER_READY.value))

    wait_for(wait, data_store, timeout)


def rollout_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    if (
        graph_manager.agent_params.algorithm.distributed_coach_synchronization_type
        == DistributedCoachSynchronizationType.SYNC
    ):
        timeout = float("inf")
    else:
        timeout = None

    # this could probably be moved up into coach.py
    graph_manager.create_graph(task_parameters)

    data_store.load_policy(graph_manager, require_new_policy=False, timeout=60)

    with graph_manager.phase_context(RunPhase.TRAIN):
        # this worker should play a fraction of the total playing steps per rollout
        graph_manager.reset_internal_state(force_environment_reset=True)

        act_steps = (
            graph_manager.agent_params.algorithm.num_consecutive_playing_steps
            / num_workers
        )
        for i in range(graph_manager.improve_steps / act_steps):
            if data_store.end_of_policies():
                break

            graph_manager.act(
                act_steps,
                wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes,
            )

            data_store.load_policy(graph_manager, require_new_policy=True, timeout=timeout)
