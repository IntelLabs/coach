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
import math
from collections import defaultdict

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType
from rl_coach.checkpoint import CheckpointStateFile, CheckpointStateReader
from rl_coach.core_types import EnvironmentSteps, RunPhase, EnvironmentEpisodes
from rl_coach.data_stores.data_store import SyncFiles


def wait_for_checkpoint(checkpoint_directory, data_store=None, timeout=10):
    """
    block until there is a checkpoint in checkpoint_directory
    """
    chkpt_state_file = CheckpointStateFile(checkpoint_directory)
    for i in range(timeout):
        if data_store:
            data_store.load_from_store()

        if chkpt_state_file.read() is not None:
            return
        time.sleep(10)

    # one last time
    if chkpt_state_file.read() is not None:
        return

    raise ValueError((
        'Waited {timeout} seconds, but checkpoint never found in '
        '{checkpoint_directory}'
    ).format(
        timeout=timeout,
        checkpoint_directory=checkpoint_directory,
    ))


def should_stop(checkpoint_directory):
    return os.path.exists(os.path.join(checkpoint_directory, SyncFiles.FINISHED.value))


last_checkpoints = defaultdict(int)


def new_checkpoint_exists(checkpoint_directory, data_store, wait):
    checkpoint_state_reader = CheckpointStateReader(checkpoint_directory, checkpoint_state_optional=False)
    checkpoint = None
    while wait or checkpoint is None:
        if should_stop(checkpoint_directory):
            return False
        if data_store:
            data_store.load_from_store()
        checkpoint = checkpoint_state_reader.get_latest()
        if checkpoint is not None and checkpoint.num > last_checkpoints[checkpoint_directory]:
            last_checkpoints[checkpoint_directory] = checkpoint.num
            return True

    return False


def rollout_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    checkpoint_directory = task_parameters.checkpoint_restore_path
    wait_for_checkpoint(checkpoint_directory, data_store)

    wait = graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC

    graph_manager.create_graph(task_parameters)

    with graph_manager.phase_context(RunPhase.TRAIN):
        # this worker should play a fraction of the total playing steps per rollout
        act_steps = graph_manager.agent_params.algorithm.num_consecutive_playing_steps / num_workers
        for i in range(graph_manager.improve_steps / act_steps):
            if should_stop(checkpoint_directory):
                break

            graph_manager.act(act_steps, wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes)

            if new_checkpoint_exists(checkpoint_directory, data_store, wait):
                graph_manager.restore_checkpoint()
