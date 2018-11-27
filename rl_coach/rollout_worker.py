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

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType
from rl_coach.checkpoint import CheckpointStateFile, CheckpointStateReader
from rl_coach.core_types import EnvironmentSteps, RunPhase, EnvironmentEpisodes
from rl_coach.data_stores.data_store import SyncFiles


def wait_for_checkpoint(checkpoint_dir, data_store=None, timeout=10):
    """
    block until there is a checkpoint in checkpoint_dir
    """
    chkpt_state_file = CheckpointStateFile(checkpoint_dir)
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
        '{checkpoint_dir}'
    ).format(
        timeout=timeout,
        checkpoint_dir=checkpoint_dir,
    ))


def should_stop(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, SyncFiles.FINISHED.value))


def rollout_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    checkpoint_dir = task_parameters.checkpoint_restore_dir
    wait_for_checkpoint(checkpoint_dir, data_store)

    graph_manager.create_graph(task_parameters)
    with graph_manager.phase_context(RunPhase.TRAIN):

        chkpt_state_reader = CheckpointStateReader(checkpoint_dir, checkpoint_state_optional=False)
        last_checkpoint = 0

        act_steps = math.ceil((graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps)/num_workers)

        for i in range(int(graph_manager.improve_steps.num_steps/act_steps)):

            if should_stop(checkpoint_dir):
                break

            if type(graph_manager.agent_params.algorithm.num_consecutive_playing_steps) == EnvironmentSteps:
                graph_manager.act(EnvironmentSteps(num_steps=act_steps), wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes)
            elif type(graph_manager.agent_params.algorithm.num_consecutive_playing_steps) == EnvironmentEpisodes:
                graph_manager.act(EnvironmentEpisodes(num_steps=act_steps))

            new_checkpoint = chkpt_state_reader.get_latest()
            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                while new_checkpoint is None or new_checkpoint.num < last_checkpoint + 1:
                    if should_stop(checkpoint_dir):
                        break
                    if data_store:
                        data_store.load_from_store()
                    new_checkpoint = chkpt_state_reader.get_latest()

                graph_manager.restore_checkpoint()

            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.ASYNC:
                if new_checkpoint is not None and new_checkpoint.num > last_checkpoint:
                    graph_manager.restore_checkpoint()

            if new_checkpoint is not None:
                last_checkpoint = new_checkpoint.num
