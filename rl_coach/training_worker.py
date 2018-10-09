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

def training_worker(graph_manager, checkpoint_dir, policy_type):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    # initialize graph
    task_parameters = TaskParameters()
    task_parameters.__dict__['save_checkpoint_dir'] = checkpoint_dir
    task_parameters.__dict__['save_checkpoint_secs'] = 60
    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # training loop
    steps = 0

    # evaluation offset
    eval_offset = 1

    while(steps < graph_manager.improve_steps.num_steps):
        if graph_manager.should_train():
            steps += 1

            graph_manager.phase = core_types.RunPhase.TRAIN
            graph_manager.train(core_types.TrainingSteps(1))
            graph_manager.phase = core_types.RunPhase.UNDEFINED

            if steps * graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps > graph_manager.steps_between_evaluation_periods.num_steps * eval_offset:
                graph_manager.evaluate(graph_manager.evaluation_steps)
                eval_offset += 1

            if policy_type == 'ON':
                graph_manager.save_checkpoint()
            else:
                graph_manager.occasionally_save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint-dir',
                        help='(string) Path to a folder containing a checkpoint to write the model to.',
                        type=str,
                        default='/checkpoint')
    parser.add_argument('--memory-backend-params',
                        help="(string) JSON string of the memory backend params",
                        type=str)
    parser.add_argument('--data-store-params',
                        help="(string) JSON string of the data store params",
                        type=str)
    parser.add_argument('--policy-type',
                        help="(string) The type of policy: OFF/ON",
                        type=str,
                        default='OFF')
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
        policy_type=args.policy_type
    )


if __name__ == '__main__':
    main()
