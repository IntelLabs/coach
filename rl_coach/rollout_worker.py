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
from rl_coach.core_types import EnvironmentSteps, RunPhase, EnvironmentEpisodes
from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from rl_coach.data_stores.data_store import SyncFiles


def has_checkpoint(checkpoint_dir):
    """
    True if a checkpoint is present in checkpoint_dir
    """
    if os.path.isdir(checkpoint_dir):
        if len(os.listdir(checkpoint_dir)) > 0:
            return os.path.isfile(os.path.join(checkpoint_dir, "checkpoint"))

    return False


def wait_for_checkpoint(checkpoint_dir, data_store=None, timeout=10):
    """
    block until there is a checkpoint in checkpoint_dir
    """
    for i in range(timeout):
        if data_store:
            data_store.load_from_store()

        if has_checkpoint(checkpoint_dir):
            return
        time.sleep(10)

    # one last time
    if has_checkpoint(checkpoint_dir):
        return

    raise ValueError((
        'Waited {timeout} seconds, but checkpoint never found in '
        '{checkpoint_dir}'
    ).format(
        timeout=timeout,
        checkpoint_dir=checkpoint_dir,
    ))


def data_store_ckpt_load(data_store):
    while True:
        data_store.load_from_store()
        time.sleep(10)


def get_latest_checkpoint(checkpoint_dir):
    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint')):
        ckpt = CheckpointState()
        contents = open(os.path.join(checkpoint_dir, 'checkpoint'), 'r').read()
        text_format.Merge(contents, ckpt)
        rel_path = os.path.relpath(ckpt.model_checkpoint_path, checkpoint_dir)
        return int(rel_path.split('_Step')[0])

    return 0


def should_stop(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, SyncFiles.FINISHED.value))


def rollout_worker(graph_manager, checkpoint_dir, data_store, num_workers):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    wait_for_checkpoint(checkpoint_dir)

    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_restore_dir'] = checkpoint_dir

    graph_manager.create_graph(task_parameters)
    with graph_manager.phase_context(RunPhase.TRAIN):

        last_checkpoint = 0

        act_steps = math.ceil((graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps)/num_workers)

        for i in range(int(graph_manager.improve_steps.num_steps/act_steps)):

            if should_stop(checkpoint_dir):
                break

            if type(graph_manager.agent_params.algorithm.num_consecutive_playing_steps) == EnvironmentSteps:
                graph_manager.act(EnvironmentSteps(num_steps=act_steps), wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes)
            elif type(graph_manager.agent_params.algorithm.num_consecutive_playing_steps) == EnvironmentEpisodes:
                graph_manager.act(EnvironmentEpisodes(num_steps=act_steps))

            new_checkpoint = get_latest_checkpoint(checkpoint_dir)

            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                while new_checkpoint < last_checkpoint + 1:
                    if should_stop(checkpoint_dir):
                        break
                    if data_store:
                        data_store.load_from_store()
                    new_checkpoint = get_latest_checkpoint(checkpoint_dir)

                graph_manager.restore_checkpoint()

            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.ASYNC:
                if new_checkpoint > last_checkpoint:
                    graph_manager.restore_checkpoint()

            last_checkpoint = new_checkpoint
