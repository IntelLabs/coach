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

import sys
sys.path.append('.')

import copy
from rl_coach.core_types import EnvironmentSteps
import os
from rl_coach import logger
import traceback
from rl_coach.logger import screen, failed_imports
import argparse
import atexit
import time
import sys
from rl_coach.base_parameters import Frameworks, VisualizationParameters, TaskParameters, DistributedTaskParameters
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import subprocess
from rl_coach.graph_managers.graph_manager import HumanPlayScheduleParameters, GraphManager
from rl_coach.utils import list_all_presets, short_dynamic_import, get_open_port, SharedMemoryScratchPad, get_base_dir
from rl_coach.agents.human_agent import HumanAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.environment import SingleLevelSelection


if len(set(failed_imports)) > 0:
    screen.warning("Warning: failed to import the following packages - {}".format(', '.join(set(failed_imports))))


def get_graph_manager_from_args(args: argparse.Namespace) -> 'GraphManager':
    """
    Return the graph manager according to the command line arguments given by the user
    :param args: the arguments given by the user
    :return: the updated graph manager
    """

    graph_manager = None

    # if a preset was given we will load the graph manager for the preset
    if args.preset is not None:
        graph_manager = short_dynamic_import(args.preset, ignore_module_case=True)

    # for human play we need to create a custom graph manager
    if args.play:
        env_params = short_dynamic_import(args.environment_type, ignore_module_case=True)()
        env_params.human_control = True
        schedule_params = HumanPlayScheduleParameters()
        graph_manager = BasicRLGraphManager(HumanAgentParameters(), env_params, schedule_params, VisualizationParameters())

    if args.level:
        if isinstance(graph_manager.env_params.level, SingleLevelSelection):
            graph_manager.env_params.level.select(args.level)
        else:
            graph_manager.env_params.level = args.level

    # set the seed for the environment
    if args.seed is not None:
        graph_manager.env_params.seed = args.seed

    # visualization
    graph_manager.visualization_parameters.dump_gifs = graph_manager.visualization_parameters.dump_gifs or args.dump_gifs
    graph_manager.visualization_parameters.dump_mp4 = graph_manager.visualization_parameters.dump_mp4 or args.dump_mp4
    graph_manager.visualization_parameters.render = args.render
    graph_manager.visualization_parameters.tensorboard = args.tensorboard

    # update the custom parameters
    if args.custom_parameter is not None:
        unstripped_key_value_pairs = [pair.split('=') for pair in args.custom_parameter.split(';')]
        stripped_key_value_pairs = [tuple([pair[0].strip(), pair[1].strip()]) for pair in
                                    unstripped_key_value_pairs if len(pair) == 2]

        # load custom parameters into run_dict
        for key, value in stripped_key_value_pairs:
            exec("graph_manager.{}={}".format(key, value))

    return graph_manager


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse the arguments that the user entered
    :param parser: the argparse command line parser
    :return: the parsed arguments
    """
    args = parser.parse_args()

    # if no arg is given
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    # list available presets
    preset_names = list_all_presets()
    if args.list:
        screen.log_title("Available Presets:")
        for preset in sorted(preset_names):
            print(preset)
        sys.exit(0)

    # replace a short preset name with the full path
    if args.preset is not None:
        if args.preset.lower() in [p.lower() for p in preset_names]:
            args.preset = "{}.py:graph_manager".format(os.path.join(get_base_dir(), 'presets', args.preset))
        else:
            args.preset = "{}".format(args.preset)
            # if a graph manager variable was not specified, try the default of :graph_manager
            if len(args.preset.split(":")) == 1:
                args.preset += ":graph_manager"

        # verify that the preset exists
        preset_path = args.preset.split(":")[0]
        if not os.path.exists(preset_path):
            screen.error("The given preset ({}) cannot be found.".format(args.preset))

        # verify that the preset can be instantiated
        try:
            short_dynamic_import(args.preset, ignore_module_case=True)
        except TypeError as e:
            traceback.print_exc()
            screen.error('Internal Error: ' + str(e) + "\n\nThe given preset ({}) cannot be instantiated."
                         .format(args.preset))

    # validate the checkpoints args
    if args.checkpoint_restore_dir is not None and not os.path.exists(args.checkpoint_restore_dir):
        screen.error("The requested checkpoint folder to load from does not exist.")

    # no preset was given. check if the user requested to play some environment on its own
    if args.preset is None and args.play:
        if args.environment_type:
            args.agent_type = 'Human'
        else:
            screen.error('When no preset is given for Coach to run, and the user requests human control over '
                         'the environment, the user is expected to input the desired environment_type and level.'
                         '\nAt least one of these parameters was not given.')
    elif args.preset and args.play:
        screen.error("Both the --preset and the --play flags were set. These flags can not be used together. "
                     "For human control, please use the --play flag together with the environment type flag (-et)")
    elif args.preset is None and not args.play:
        screen.error("Please choose a preset using the -p flag or use the --play flag together with choosing an "
                     "environment type (-et) in order to play the game.")

    # get experiment name and path
    args.experiment_name = logger.get_experiment_name(args.experiment_name)
    args.experiment_path = logger.get_experiment_path(args.experiment_name)

    if args.play and args.num_workers > 1:
        screen.warning("Playing the game as a human is only available with a single worker. "
                       "The number of workers will be reduced to 1")
        args.num_workers = 1

    args.framework = Frameworks[args.framework.lower()]

    # checkpoints
    args.save_checkpoint_dir = os.path.join(args.experiment_path, 'checkpoint') if args.save_checkpoint_secs is not None else None

    return args


def add_items_to_dict(target_dict, source_dict):
    updated_task_parameters = copy.copy(source_dict)
    updated_task_parameters.update(target_dict)
    return updated_task_parameters


def open_dashboard(experiment_path):
    dashboard_path = 'python {}/dashboard.py'.format(get_base_dir())
    cmd = "{} --experiment_dir {}".format(dashboard_path, experiment_path)
    screen.log_title("Opening dashboard - experiment path: {}".format(experiment_path))
    # subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, executable="/bin/bash")
    subprocess.Popen(cmd, shell=True, executable="/bin/bash")


def start_graph(graph_manager: 'GraphManager', task_parameters: 'TaskParameters'):
    graph_manager.create_graph(task_parameters)

    # let the adventure begin
    if task_parameters.evaluate_only:
        graph_manager.evaluate(EnvironmentSteps(sys.maxsize), keep_networks_in_sync=True)
    else:
        graph_manager.improve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        default=None,
                        type=str)
    parser.add_argument('-l', '--list',
                        help="(flag) List all available presets",
                        action='store_true')
    parser.add_argument('-e', '--experiment_name',
                        help="(string) Experiment name to be used to store the results.",
                        default='',
                        type=str)
    parser.add_argument('-r', '--render',
                        help="(flag) Render environment",
                        action='store_true')
    parser.add_argument('-f', '--framework',
                        help="(string) Neural network framework. Available values: tensorflow",
                        default='tensorflow',
                        type=str)
    parser.add_argument('-n', '--num_workers',
                        help="(int) Number of workers for multi-process based agents, e.g. A3C",
                        default=1,
                        type=int)
    parser.add_argument('-c', '--use_cpu',
                        help="(flag) Use only the cpu for training. If a GPU is not available, this flag will have no "
                             "effect and the CPU will be used either way.",
                        action='store_true')
    parser.add_argument('-ew', '--evaluation_worker',
                        help="(int) If multiple workers are used, add an evaluation worker as well which will "
                             "evaluate asynchronously and independently during the training. NOTE: this worker will "
                             "ignore the evaluation settings in the preset's ScheduleParams.",
                        action='store_true')
    parser.add_argument('--play',
                        help="(flag) Play as a human by controlling the game with the keyboard. "
                             "This option will save a replay buffer with the game play.",
                        action='store_true')
    parser.add_argument('--evaluate',
                        help="(flag) Run evaluation only. This is a convenient way to disable "
                             "training in order to evaluate an existing checkpoint.",
                        action='store_true')
    parser.add_argument('-v', '--verbosity',
                        help="(flag) Sets the verbosity level of Coach print outs. Can be either low or high.",
                        default="low",
                        type=str)
    parser.add_argument('-tfv', '--tf_verbosity',
                        help="(flag) TensorFlow verbosity level",
                        default=3,
                        type=int)
    parser.add_argument('-s', '--save_checkpoint_secs',
                        help="(int) Time in seconds between saving checkpoints of the model.",
                        default=None,
                        type=int)
    parser.add_argument('-crd', '--checkpoint_restore_dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str)
    parser.add_argument('-dg', '--dump_gifs',
                        help="(flag) Enable the gif saving functionality.",
                        action='store_true')
    parser.add_argument('-dm', '--dump_mp4',
                        help="(flag) Enable the mp4 saving functionality.",
                        action='store_true')
    parser.add_argument('-at', '--agent_type',
                        help="(string) Choose an agent type class to override on top of the selected preset. "
                             "If no preset is defined, a preset can be set from the command-line by combining settings "
                             "which are set by using --agent_type, --experiment_type, --environemnt_type",
                        default=None,
                        type=str)
    parser.add_argument('-et', '--environment_type',
                        help="(string) Choose an environment type class to override on top of the selected preset."
                             "If no preset is defined, a preset can be set from the command-line by combining settings "
                             "which are set by using --agent_type, --experiment_type, --environemnt_type",
                        default=None,
                        type=str)
    parser.add_argument('-ept', '--exploration_policy_type',
                        help="(string) Choose an exploration policy type class to override on top of the selected "
                             "preset."
                             "If no preset is defined, a preset can be set from the command-line by combining settings "
                             "which are set by using --agent_type, --experiment_type, --environemnt_type"
                        ,
                        default=None,
                        type=str)
    parser.add_argument('-lvl', '--level',
                        help="(string) Choose the level that will be played in the environment that was selected."
                             "This value will override the level parameter in the environment class."
                        ,
                        default=None,
                        type=str)
    parser.add_argument('-cp', '--custom_parameter',
                        help="(string) Semicolon separated parameters used to override specific parameters on top of"
                             " the selected preset (or on top of the command-line assembled one). "
                             "Whenever a parameter value is a string, it should be inputted as '\\\"string\\\"'. "
                             "For ex.: "
                             "\"visualization.render=False; num_training_iterations=500; optimizer='rmsprop'\"",
                        default=None,
                        type=str)
    parser.add_argument('--print_parameters',
                        help="(flag) Print tuning_parameters to stdout",
                        action='store_true')
    parser.add_argument('-tb', '--tensorboard',
                        help="(flag) When using the TensorFlow backend, enable TensorBoard log dumps. ",
                        action='store_true')
    parser.add_argument('-ns', '--no_summary',
                        help="(flag) Prevent Coach from printing a summary and asking questions at the end of runs",
                        action='store_true')
    parser.add_argument('-d', '--open_dashboard',
                        help="(flag) Open dashboard with the experiment when the run starts",
                        action='store_true')
    parser.add_argument('--seed',
                        help="(int) A seed to use for running the experiment",
                        default=None,
                        type=int)

    args = parse_arguments(parser)

    graph_manager = get_graph_manager_from_args(args)

    # Intel optimized TF seems to run significantly faster when limiting to a single OMP thread.
    # This will not affect GPU runs.
    os.environ["OMP_NUM_THREADS"] = "1"

    # turn TF debug prints off
    if args.framework == Frameworks.tensorflow:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_verbosity)

    # turn off the summary at the end of the run if necessary
    if not args.no_summary:
        atexit.register(logger.summarize_experiment)
        screen.change_terminal_title(args.experiment_name)

    # open dashboard
    if args.open_dashboard:
        open_dashboard(args.experiment_path)

    # Single-threaded runs
    if args.num_workers == 1:
        # Start the training or evaluation
        task_parameters = TaskParameters(framework_type="tensorflow",  # TODO: tensorflow should'nt be hardcoded
                                         evaluate_only=args.evaluate,
                                         experiment_path=args.experiment_path,
                                         seed=args.seed,
                                         use_cpu=args.use_cpu)
        task_parameters.__dict__ = add_items_to_dict(task_parameters.__dict__, args.__dict__)

        start_graph(graph_manager=graph_manager, task_parameters=task_parameters)

    # Multi-threaded runs
    else:
        total_tasks = args.num_workers
        if args.evaluation_worker:
            total_tasks += 1

        ps_hosts = "localhost:{}".format(get_open_port())
        worker_hosts = ",".join(["localhost:{}".format(get_open_port()) for i in range(total_tasks)])

        # Shared memory
        class CommManager(BaseManager):
            pass
        CommManager.register('SharedMemoryScratchPad', SharedMemoryScratchPad, exposed=['add', 'get', 'internal_call'])
        comm_manager = CommManager()
        comm_manager.start()
        shared_memory_scratchpad = comm_manager.SharedMemoryScratchPad()

        def start_distributed_task(job_type, task_index, evaluation_worker=False,
                                   shared_memory_scratchpad=shared_memory_scratchpad):
            task_parameters = DistributedTaskParameters(framework_type="tensorflow", # TODO: tensorflow should'nt be hardcoded
                                                        parameters_server_hosts=ps_hosts,
                                                        worker_hosts=worker_hosts,
                                                        job_type=job_type,
                                                        task_index=task_index,
                                                        evaluate_only=evaluation_worker,
                                                        use_cpu=args.use_cpu,
                                                        num_tasks=total_tasks,  # training tasks + 1 evaluation task
                                                        num_training_tasks=args.num_workers,
                                                        experiment_path=args.experiment_path,
                                                        shared_memory_scratchpad=shared_memory_scratchpad,
                                                        seed=args.seed+task_index if args.seed is not None else None)  # each worker gets a different seed
            task_parameters.__dict__ = add_items_to_dict(task_parameters.__dict__, args.__dict__)
            # we assume that only the evaluation workers are rendering
            graph_manager.visualization_parameters.render = args.render and evaluation_worker
            p = Process(target=start_graph, args=(graph_manager, task_parameters))
            # p.daemon = True
            p.start()
            return p

        # parameter server
        parameter_server = start_distributed_task("ps", 0)

        # training workers
        # wait a bit before spawning the non chief workers in order to make sure the session is already created
        workers = []
        workers.append(start_distributed_task("worker", 0))
        time.sleep(2)
        for task_index in range(1, args.num_workers):
            workers.append(start_distributed_task("worker", task_index))

        # evaluation worker
        if args.evaluation_worker:
            evaluation_worker = start_distributed_task("worker", args.num_workers, evaluation_worker=True)

        # wait for all workers
        [w.join() for w in workers]
        if args.evaluation_worker:
            evaluation_worker.terminate()


if __name__ == "__main__":
    main()
