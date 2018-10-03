"""
"""
import argparse
import time
import json

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach import core_types
from rl_coach.utils import short_dynamic_import
from rl_coach.memories.backend.memory_impl import construct_memory_params

# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take a non-distributed preset and automatically distribute it.


def training_worker(graph_manager, checkpoint_dir):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    # initialize graph
    task_parameters = TaskParameters()
    task_parameters.__dict__['save_checkpoint_dir'] = checkpoint_dir
    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # training loop
    while True:
        graph_manager.phase = core_types.RunPhase.TRAIN
        graph_manager.train(core_types.TrainingSteps(1))
        graph_manager.phase = core_types.RunPhase.UNDEFINED

        graph_manager.evaluate(graph_manager.evaluation_steps)

        graph_manager.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint_dir',
                        help='(string) Path to a folder containing a checkpoint to write the model to.',
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
            args.memory_backend_params['run_type'] = 'trainer'
        graph_manager.agent_params.memory.register_var('memory_backend_params', construct_memory_params(args.memory_backend_params))

    training_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
