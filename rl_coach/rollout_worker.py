"""
this rollout worker:

- restores a model from disk
- evaluates a predefined number of episodes
- contributes them to a distributed memory
- exits
"""

import argparse
import time
import os
import json
import math

from threading import Thread

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach.core_types import EnvironmentSteps, RunPhase
from rl_coach.utils import short_dynamic_import
from rl_coach.memories.backend.memory_impl import construct_memory_params
from rl_coach.data_stores.data_store_impl import get_data_store, construct_data_store_params
from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take
#    a non-distributed preset and automatically distribute it.

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


def rollout_worker(graph_manager, checkpoint_dir, data_store, num_workers, policy_type):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    wait_for_checkpoint(checkpoint_dir)

    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_restore_dir'] = checkpoint_dir
    time.sleep(30)
    graph_manager.create_graph(task_parameters)
    with graph_manager.phase_context(RunPhase.TRAIN):
        error_compensation = 100

        last_checkpoint = 0

        act_steps = math.ceil((graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps + error_compensation)/num_workers)

        for i in range(int(graph_manager.improve_steps.num_steps/act_steps)):

            graph_manager.act(EnvironmentSteps(num_steps=act_steps))

            new_checkpoint = get_latest_checkpoint(checkpoint_dir)

            if policy_type == 'ON':
                while new_checkpoint < last_checkpoint + 1:
                    if data_store:
                        data_store.load_from_store()
                    new_checkpoint = get_latest_checkpoint(checkpoint_dir)

                graph_manager.restore_checkpoint()

            if policy_type == "OFF":

                if new_checkpoint > last_checkpoint:
                    graph_manager.restore_checkpoint()

            last_checkpoint = new_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint-dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str,
                        default='/checkpoint')
    parser.add_argument('--memory-backend-params',
                        help="(string) JSON string of the memory backend params",
                        type=str)
    parser.add_argument('--data-store-params',
                        help="(string) JSON string of the data store params",
                        type=str)
    parser.add_argument('--num-workers',
                        help="(int) The number of workers started in this pool",
                        type=int,
                        default=1)
    parser.add_argument('--policy-type',
                        help="(string) The type of policy: OFF/ON",
                        type=str,
                        default='OFF')

    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    data_store = None
    if args.memory_backend_params:
        args.memory_backend_params = json.loads(args.memory_backend_params)
        args.memory_backend_params['run_type'] = 'worker'
        graph_manager.agent_params.memory.register_var('memory_backend_params', construct_memory_params(args.memory_backend_params))

    if args.data_store_params:
        data_store_params = construct_data_store_params(json.loads(args.data_store_params))
        data_store_params.checkpoint_dir = args.checkpoint_dir
        graph_manager.data_store_params = data_store_params
        data_store = get_data_store(data_store_params)
        wait_for_checkpoint(checkpoint_dir=args.checkpoint_dir, data_store=data_store)
        # thread = Thread(target = data_store_ckpt_load, args = [data_store])
        # thread.start()

    rollout_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
        data_store=data_store,
        num_workers=args.num_workers,
        policy_type=args.policy_type
    )

if __name__ == '__main__':
    main()
