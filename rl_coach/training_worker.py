"""
"""
import argparse
import time
import json

from threading import Thread

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach import core_types
from rl_coach.utils import short_dynamic_import
from rl_coach.memories.backend.memory_impl import construct_memory_params
from rl_coach.data_stores.data_store_impl import get_data_store, construct_data_store_params

def data_store_ckpt_save(data_store):
    while True:
        data_store.save_to_store()
        time.sleep(10)

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
    parser.add_argument('--data_store_params',
                        help="(string) JSON string of the data store params",
                        type=str)
    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    if args.memory_backend_params:
        args.memory_backend_params = json.loads(args.memory_backend_params)
        args.memory_backend_params['run_type'] = 'trainer'
        graph_manager.agent_params.memory.register_var('memory_backend_params', construct_memory_params(args.memory_backend_params))

    if args.data_store_params:
        data_store_params = construct_data_store_params(json.loads(args.data_store_params))
        data_store_params.checkpoint_dir = args.checkpoint_dir
        graph_manager.data_store_params = data_store_params
        # data_store = get_data_store(data_store_params)
        # thread = Thread(target = data_store_ckpt_save, args = [data_store])
        # thread.start()

    training_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
