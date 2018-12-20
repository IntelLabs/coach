"""
this rollout worker:

- restores a model from disk
- evaluates the restored model
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
from rl_coach.rollout_worker import wait_for_checkpoint, should_stop
from rl_coach.logger import screen


def eval_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform evaluation using the model
    """
    checkpoint_dir = task_parameters.checkpoint_restore_dir
    wait_for_checkpoint(checkpoint_dir, data_store)

    graph_manager.create_graph(task_parameters)
    with graph_manager.phase_context(RunPhase.TRAIN):

        chkpt_state_reader = CheckpointStateReader(checkpoint_dir, checkpoint_state_optional=False)
        last_checkpoint = 0

        act_steps = math.ceil((graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps)/num_workers)
        # graph_manager.evaluate(graph_manager.evaluation_steps)
        for i in range(int(graph_manager.improve_steps.num_steps/act_steps)):

            if should_stop(task_parameters.checkpoint_save_dir):
                break
            
            if graph_manager.evaluate(graph_manager.evaluation_steps):
                data_store.save_finished_to_store()
                break
                    
            new_checkpoint = chkpt_state_reader.get_latest()
            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                while new_checkpoint is None or new_checkpoint.num < last_checkpoint + 1:
                    if should_stop(task_parameters.checkpoint_save_dir):
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


