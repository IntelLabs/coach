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

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach.core_types import EnvironmentEpisodes, RunPhase
from rl_coach.utils import short_dynamic_import
from rl_coach.memories.backend.memory_impl import construct_memory_params


# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take
#    a non-distributed preset and automatically distribute it.

def has_checkpoint(checkpoint_dir):
    """
    True if a checkpoint is present in checkpoint_dir
    """
    return len(os.listdir(checkpoint_dir)) > 0


def wait_for_checkpoint(checkpoint_dir, timeout=10):
    """
    block until there is a checkpoint in checkpoint_dir
    """
    for i in range(timeout):
        if has_checkpoint(checkpoint_dir):
            return
        time.sleep(1)

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


def rollout_worker(graph_manager, checkpoint_dir):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    wait_for_checkpoint(checkpoint_dir)

    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_restore_dir'] = checkpoint_dir
    graph_manager.create_graph(task_parameters)
    graph_manager.phase = RunPhase.TRAIN

    for i in range(10000000):
        graph_manager.act(EnvironmentEpisodes(num_steps=10))
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

    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    if args.memory_backend_params:
        args.memory_backend_params = json.loads(args.memory_backend_params)
        if 'run_type' not in args.memory_backend_params:
            args.memory_backend_params['run_type'] = 'worker'
        graph_manager.agent_params.memory.register_var('memory_backend_params', construct_memory_params(args.memory_backend_params))
    rollout_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )

if __name__ == '__main__':
    main()
