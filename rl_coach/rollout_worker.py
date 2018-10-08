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


def check_for_new_checkpoint(checkpoint_dir, last_checkpoint):
    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint')):
        ckpt = CheckpointState()
        contents = open(os.path.join(checkpoint_dir, 'checkpoint'), 'r').read()
        text_format.Merge(contents, ckpt)
        rel_path = os.path.relpath(ckpt.model_checkpoint_path, checkpoint_dir)
        current_checkpoint = int(rel_path.split('_Step')[0])
        if current_checkpoint > last_checkpoint:
            last_checkpoint = current_checkpoint

        return last_checkpoint


def rollout_worker(graph_manager, checkpoint_dir, data_store):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    wait_for_checkpoint(checkpoint_dir)

    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_restore_dir'] = checkpoint_dir
    time.sleep(30)
    graph_manager.create_graph(task_parameters)
    graph_manager.phase = RunPhase.TRAIN

    error_compensation = 100

    last_checkpoint = 0

    act_steps = graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps + error_compensation

    print(act_steps, graph_manager.improve_steps.num_steps)

    for i in range(int(graph_manager.improve_steps.num_steps/act_steps)):

        graph_manager.act(EnvironmentSteps(num_steps=act_steps))

        new_checkpoint = last_checkpoint + 1
        while last_checkpoint < new_checkpoint:
            if data_store:
                data_store.load_from_store()
            last_checkpoint = check_for_new_checkpoint(checkpoint_dir, last_checkpoint)

        last_checkpoint = new_checkpoint
        graph_manager.restore_checkpoint()

    graph_manager.phase = RunPhase.UNDEFINED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint_dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str,
                        default='/checkpoint')
    parser.add_argument('-r', '--redis_ip',
                        help="(string) IP or host for the redis server",
                        default='localhost',
                        type=str)
    parser.add_argument('-rp', '--redis_port',
                        help="(int) Port of the redis server",
                        default=6379,
                        type=int)
    parser.add_argument('--memory_backend_params',
                        help="(string) JSON string of the memory backend params",
                        type=str)
    parser.add_argument('--data_store_params',
                        help="(string) JSON string of the data store params",
                        type=str)

    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    data_store = None
    if args.memory_backend_params:
        args.memory_backend_params = json.loads(args.memory_backend_params)
        print(args.memory_backend_params)
        args.memory_backend_params['run_type'] = 'worker'
        print(construct_memory_params(args.memory_backend_params))
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
        data_store=data_store
    )

if __name__ == '__main__':
    main()
