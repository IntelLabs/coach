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
from configparser import ConfigParser, Error
from rl_coach.core_types import EnvironmentSteps
import os
from rl_coach import logger
import traceback
from rl_coach.logger import screen, failed_imports
import argparse
import atexit
import time
import sys
import json
from rl_coach.base_parameters import Frameworks, VisualizationParameters, TaskParameters, DistributedTaskParameters, \
    RunType, DistributedCoachSynchronizationType
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import subprocess
from rl_coach.graph_managers.graph_manager import HumanPlayScheduleParameters, GraphManager
from rl_coach.utils import list_all_presets, short_dynamic_import, get_open_port, SharedMemoryScratchPad, get_base_dir
from rl_coach.agents.human_agent import HumanAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.memories.backend.memory_impl import construct_memory_params
from rl_coach.data_stores.data_store import DataStoreParameters
from rl_coach.data_stores.s3_data_store import S3DataStoreParameters
from rl_coach.data_stores.nfs_data_store import NFSDataStoreParameters
from rl_coach.data_stores.data_store_impl import get_data_store, construct_data_store_params
from rl_coach.training_worker import training_worker
from rl_coach.rollout_worker import rollout_worker, wait_for_checkpoint


if len(set(failed_imports)) > 0:
    screen.warning("Warning: failed to import the following packages - {}".format(', '.join(set(failed_imports))))


def add_items_to_dict(target_dict, source_dict):
    updated_task_parameters = copy.copy(source_dict)
    updated_task_parameters.update(target_dict)
    return updated_task_parameters


def open_dashboard(experiment_path):
    """
    open X11 based dashboard in a new process (nonblocking)
    """
    dashboard_path = 'python {}/dashboard.py'.format(get_base_dir())
    cmd = "{} --experiment_dir {}".format(dashboard_path, experiment_path)
    screen.log_title("Opening dashboard - experiment path: {}".format(experiment_path))
    # subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, executable="/bin/bash")
    subprocess.Popen(cmd, shell=True, executable="/bin/bash")


def start_graph(graph_manager: 'GraphManager', task_parameters: 'TaskParameters'):
    """
    Runs the graph_manager using the configured task_parameters.
    This stand-alone method is a convenience for multiprocessing.
    """
    graph_manager.create_graph(task_parameters)

    # let the adventure begin
    if task_parameters.evaluate_only:
        graph_manager.evaluate(EnvironmentSteps(sys.maxsize))
    else:
        graph_manager.improve()
    graph_manager.close()


def handle_distributed_coach_tasks(graph_manager, args, task_parameters):
    ckpt_inside_container = "/checkpoint"

    memory_backend_params = None
    if args.memory_backend_params:
        memory_backend_params = json.loads(args.memory_backend_params)
        memory_backend_params['run_type'] = str(args.distributed_coach_run_type)
        graph_manager.agent_params.memory.register_var('memory_backend_params', construct_memory_params(memory_backend_params))

    data_store_params = None
    if args.data_store_params:
        data_store_params = construct_data_store_params(json.loads(args.data_store_params))
        data_store_params.expt_dir = args.experiment_path
        data_store_params.checkpoint_dir = ckpt_inside_container
        graph_manager.data_store_params = data_store_params

    if args.distributed_coach_run_type == RunType.TRAINER:
        task_parameters.checkpoint_save_dir = ckpt_inside_container
        training_worker(
            graph_manager=graph_manager,
            task_parameters=task_parameters
        )

    if args.distributed_coach_run_type == RunType.ROLLOUT_WORKER:
        task_parameters.checkpoint_restore_dir = ckpt_inside_container

        data_store = None
        if args.data_store_params:
            data_store = get_data_store(data_store_params)

        rollout_worker(
            graph_manager=graph_manager,
            data_store=data_store,
            num_workers=args.num_workers,
            task_parameters=task_parameters
        )


def handle_distributed_coach_orchestrator(args):
    from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes, \
        RunTypeParameters

    ckpt_inside_container = "/checkpoint"
    arg_list = sys.argv[1:]
    try:
        i = arg_list.index('--distributed_coach_run_type')
        arg_list.pop(i)
        arg_list.pop(i)
    except ValueError:
        pass

    trainer_command = ['python3', 'rl_coach/coach.py', '--distributed_coach_run_type', str(RunType.TRAINER)] + arg_list
    rollout_command = ['python3', 'rl_coach/coach.py', '--distributed_coach_run_type', str(RunType.ROLLOUT_WORKER)] + arg_list

    if '--experiment_name' not in rollout_command:
        rollout_command = rollout_command + ['--experiment_name', args.experiment_name]

    if '--experiment_name' not in trainer_command:
        trainer_command = trainer_command + ['--experiment_name', args.experiment_name]

    memory_backend_params = None
    if args.memory_backend == "redispubsub":
        memory_backend_params = RedisPubSubMemoryBackendParameters()

    ds_params_instance = None
    if args.data_store == "s3":
        ds_params = DataStoreParameters("s3", "", "")
        ds_params_instance = S3DataStoreParameters(ds_params=ds_params, end_point=args.s3_end_point, bucket_name=args.s3_bucket_name,
                                                   creds_file=args.s3_creds_file, checkpoint_dir=ckpt_inside_container, expt_dir=args.experiment_path)
    elif args.data_store == "nfs":
        ds_params = DataStoreParameters("nfs", "kubernetes", "")
        ds_params_instance = NFSDataStoreParameters(ds_params)

    worker_run_type_params = RunTypeParameters(args.image, rollout_command, run_type=str(RunType.ROLLOUT_WORKER), num_replicas=args.num_workers)
    trainer_run_type_params = RunTypeParameters(args.image, trainer_command, run_type=str(RunType.TRAINER))

    orchestration_params = KubernetesParameters([worker_run_type_params, trainer_run_type_params],
                                                kubeconfig='~/.kube/config',
                                                memory_backend_parameters=memory_backend_params,
                                                data_store_params=ds_params_instance)
    orchestrator = Kubernetes(orchestration_params)
    if not orchestrator.setup():
        print("Could not setup.")
        return

    if orchestrator.deploy_trainer():
        print("Successfully deployed trainer.")
    else:
        print("Could not deploy trainer.")
        return

    if orchestrator.deploy_worker():
        print("Successfully deployed rollout worker(s).")
    else:
        print("Could not deploy rollout worker(s).")
        return

    if args.dump_worker_logs:
        screen.log_title("Dumping rollout worker logs in: {}".format(args.experiment_path))
        orchestrator.worker_logs(path=args.experiment_path)

    try:
        orchestrator.trainer_logs()
    except KeyboardInterrupt:
        pass

    orchestrator.undeploy()


class CoachLauncher(object):
    """
    This class is responsible for gathering all user-specified configuration options, parsing them,
    instantiating a GraphManager and then starting that GraphManager with either improve() or evaluate().
    This class is also responsible for launching multiple processes.
    It is structured so that it can be sub-classed to provide alternate mechanisms to configure and launch
    Coach jobs.

    The key entry-point for this class is the .launch() method which is expected to be called from __main__
    and handle absolutely everything for a job.
    """

    def launch(self):
        """
        Main entry point for the class, and the standard way to run coach from the command line.
        Parses command-line arguments through argparse, instantiates a GraphManager and then runs it.
        """
        parser = self.get_argument_parser()
        args = self.get_config_args(parser)
        graph_manager = self.get_graph_manager_from_args(args)
        self.run_graph_manager(graph_manager, args)

    def get_graph_manager_from_args(self, args: argparse.Namespace) -> 'GraphManager':
        """
        Return the graph manager according to the command line arguments given by the user.
        :param args: the arguments given by the user
        :return: the graph manager, not bound to task_parameters yet.
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

        # Set framework
        # Note: Some graph managers (e.g. HAC preset) create multiple agents and the attribute is called agents_params
        if hasattr(graph_manager, 'agent_params'):
            for network_parameters in graph_manager.agent_params.network_wrappers.values():
                network_parameters.framework = args.framework
        elif hasattr(graph_manager, 'agents_params'):
            for ap in graph_manager.agents_params:
                for network_parameters in ap.network_wrappers.values():
                    network_parameters.framework = args.framework

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
        graph_manager.visualization_parameters.print_networks_summary = args.print_networks_summary

        # update the custom parameters
        if args.custom_parameter is not None:
            unstripped_key_value_pairs = [pair.split('=') for pair in args.custom_parameter.split(';')]
            stripped_key_value_pairs = [tuple([pair[0].strip(), pair[1].strip()]) for pair in
                                        unstripped_key_value_pairs if len(pair) == 2]

            # load custom parameters into run_dict
            for key, value in stripped_key_value_pairs:
                exec("graph_manager.{}={}".format(key, value))

        return graph_manager

    def display_all_presets_and_exit(self):
        # list available presets
        screen.log_title("Available Presets:")
        for preset in sorted(list_all_presets()):
            print(preset)
        sys.exit(0)

    def expand_preset(self, preset):
        """
        Replace a short preset name with the full python path, and verify that it can be imported.
        """
        if preset.lower() in [p.lower() for p in list_all_presets()]:
            preset = "{}.py:graph_manager".format(os.path.join(get_base_dir(), 'presets', preset))
        else:
            preset = "{}".format(preset)
            # if a graph manager variable was not specified, try the default of :graph_manager
            if len(preset.split(":")) == 1:
                preset += ":graph_manager"

        # verify that the preset exists
        preset_path = preset.split(":")[0]
        if not os.path.exists(preset_path):
            screen.error("The given preset ({}) cannot be found.".format(preset))

        # verify that the preset can be instantiated
        try:
            short_dynamic_import(preset, ignore_module_case=True)
        except TypeError as e:
            traceback.print_exc()
            screen.error('Internal Error: ' + str(e) + "\n\nThe given preset ({}) cannot be instantiated."
                         .format(preset))

        return preset

    def get_config_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        """
        Returns a Namespace object with all the user-specified configuration options needed to launch.
        This implementation uses argparse to take arguments from the CLI, but this can be over-ridden by
        another method that gets its configuration from elsewhere.  An equivalent method however must
        return an identically structured Namespace object, which conforms to the structure defined by
        get_argument_parser.

        This method parses the arguments that the user entered, does some basic validation, and
        modification of user-specified values in short form to be more explicit.

        :param parser: a parser object which implicitly defines the format of the Namespace that
                       is expected to be returned.
        :return: the parsed arguments as a Namespace
        """
        args = parser.parse_args()

        if args.nocolor:
            screen.set_use_colors(False)

        # if no arg is given
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        # list available presets
        if args.list:
            self.display_all_presets_and_exit()

        # Read args from config file for distributed Coach.
        if args.distributed_coach and args.distributed_coach_run_type == RunType.ORCHESTRATOR:
            coach_config = ConfigParser({
                'image': '',
                'memory_backend': 'redispubsub',
                'data_store': 's3',
                's3_end_point': 's3.amazonaws.com',
                's3_bucket_name': '',
                's3_creds_file': ''
            })
            try:
                coach_config.read(args.distributed_coach_config_path)
                args.image = coach_config.get('coach', 'image')
                args.memory_backend = coach_config.get('coach', 'memory_backend')
                args.data_store = coach_config.get('coach', 'data_store')
                if args.data_store == 's3':
                    args.s3_end_point = coach_config.get('coach', 's3_end_point')
                    args.s3_bucket_name = coach_config.get('coach', 's3_bucket_name')
                    args.s3_creds_file = coach_config.get('coach', 's3_creds_file')
            except Error as e:
                screen.error("Error when reading distributed Coach config file: {}".format(e))

            if args.image == '':
                screen.error("Image cannot be empty.")

            data_store_choices = ['s3', 'nfs']
            if args.data_store not in data_store_choices:
                screen.warning("{} data store is unsupported.".format(args.data_store))
                screen.error("Supported data stores are {}.".format(data_store_choices))

            memory_backend_choices = ['redispubsub']
            if args.memory_backend not in memory_backend_choices:
                screen.warning("{} memory backend is not supported.".format(args.memory_backend))
                screen.error("Supported memory backends are {}.".format(memory_backend_choices))

            if args.data_store == 's3':
                if args.s3_bucket_name == '':
                    screen.error("S3 bucket name cannot be empty.")
                if args.s3_creds_file == '':
                    args.s3_creds_file = None

        if args.play and args.distributed_coach:
            screen.error("Playing is not supported in distributed Coach.")

        # replace a short preset name with the full path
        if args.preset is not None:
            args.preset = self.expand_preset(args.preset)

        # validate the checkpoints args
        if args.checkpoint_restore_dir is not None and not os.path.exists(args.checkpoint_restore_dir):
            screen.error("The requested checkpoint folder to load from does not exist.")

        # no preset was given. check if the user requested to play some environment on its own
        if args.preset is None and args.play and not args.environment_type:
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
        args.checkpoint_save_dir = os.path.join(args.experiment_path, 'checkpoint') if args.checkpoint_save_secs is not None else None

        if args.export_onnx_graph and not args.checkpoint_save_secs:
            screen.warning("Exporting ONNX graphs requires setting the --checkpoint_save_secs flag. "
                           "The --export_onnx_graph will have no effect.")

        return args

    def get_argument_parser(self) -> argparse.ArgumentParser:
        """
        This returns an ArgumentParser object which defines the set of options that customers are expected to supply in order
        to launch a coach job.
        """
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
                            help="(string) Neural network framework. Available values: tensorflow, mxnet",
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
        parser.add_argument('--nocolor',
                            help="(flag) Turn off color-codes in screen logging.  Ascii text only",
                            action='store_true')
        parser.add_argument('-s', '--checkpoint_save_secs',
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
        parser.add_argument('-et', '--environment_type',
                            help="(string) Choose an environment type class to override on top of the selected preset.",
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
        parser.add_argument('--print_networks_summary',
                            help="(flag) Print network summary to stdout",
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
        parser.add_argument('-onnx', '--export_onnx_graph',
                            help="(flag) Export the ONNX graph to the experiment directory. "
                                 "This will have effect only if the --checkpoint_save_secs flag is used in order to store "
                                 "checkpoints, since the weights checkpoint are needed for the ONNX graph. "
                                 "Keep in mind that this can cause major overhead on the experiment. "
                                 "Exporting ONNX graphs requires manually installing the tf2onnx package "
                                 "(https://github.com/onnx/tensorflow-onnx).",
                            action='store_true')
        parser.add_argument('-dc', '--distributed_coach',
                            help="(flag) Use distributed Coach.",
                            action='store_true')
        parser.add_argument('-dcp', '--distributed_coach_config_path',
                            help="(string) Path to config file when using distributed rollout workers."
                                 "Only distributed Coach parameters should be provided through this config file."
                                 "Rest of the parameters are provided using Coach command line options."
                                 "Used only with --distributed_coach flag."
                                 "Ignored if --distributed_coach flag is not used.",
                            type=str)
        parser.add_argument('--memory_backend_params',
                            help=argparse.SUPPRESS,
                            type=str)
        parser.add_argument('--data_store_params',
                            help=argparse.SUPPRESS,
                            type=str)
        parser.add_argument('--distributed_coach_run_type',
                            help=argparse.SUPPRESS,
                            type=RunType,
                            default=RunType.ORCHESTRATOR,
                            choices=list(RunType))
        parser.add_argument('-asc', '--apply_stop_condition',
                            help="(flag) If set, this will apply a stop condition on the run, defined by reaching a"
                                 "target success rate as set by the environment or a custom success rate as defined "
                                 "in the preset. ",
                            action='store_true')
        parser.add_argument('--dump_worker_logs',
                            help="(flag) Only used in distributed coach. If set, the worker logs are saved in the experiment dir",
                            action='store_true')

        return parser

    def run_graph_manager(self, graph_manager: 'GraphManager', args: argparse.Namespace):
        if args.distributed_coach and not graph_manager.agent_params.algorithm.distributed_coach_synchronization_type:
            screen.error("{} algorithm is not supported using distributed Coach.".format(graph_manager.agent_params.algorithm))

        if args.distributed_coach and args.checkpoint_save_secs and graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
            screen.warning("The --checkpoint_save_secs or -s argument will be ignored as SYNC distributed coach sync type is used. Checkpoint will be saved every training iteration.")

        if args.distributed_coach and not args.checkpoint_save_secs and graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.ASYNC:
            screen.error("Distributed coach with ASYNC distributed coach sync type requires --checkpoint_save_secs or -s.")

        # Intel optimized TF seems to run significantly faster when limiting to a single OMP thread.
        # This will not affect GPU runs.
        os.environ["OMP_NUM_THREADS"] = "1"

        # turn TF debug prints off
        if args.framework == Frameworks.tensorflow:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_verbosity)

        # turn off the summary at the end of the run if necessary
        if not args.no_summary and not args.distributed_coach:
            atexit.register(logger.summarize_experiment)
            screen.change_terminal_title(args.experiment_name)

        task_parameters = TaskParameters(
            framework_type=args.framework,
            evaluate_only=args.evaluate,
            experiment_path=args.experiment_path,
            seed=args.seed,
            use_cpu=args.use_cpu,
            checkpoint_save_secs=args.checkpoint_save_secs,
            checkpoint_restore_dir=args.checkpoint_restore_dir,
            checkpoint_save_dir=args.checkpoint_save_dir,
            export_onnx_graph=args.export_onnx_graph,
            apply_stop_condition=args.apply_stop_condition
        )

        # open dashboard
        if args.open_dashboard:
            open_dashboard(args.experiment_path)

        if args.distributed_coach and args.distributed_coach_run_type != RunType.ORCHESTRATOR:
            handle_distributed_coach_tasks(graph_manager, args, task_parameters)
            return

        if args.distributed_coach and args.distributed_coach_run_type == RunType.ORCHESTRATOR:
            handle_distributed_coach_orchestrator(args)
            return

        # Single-threaded runs
        if args.num_workers == 1:
            self.start_single_threaded(task_parameters, graph_manager, args)
        else:
            self.start_multi_threaded(graph_manager, args)

    def start_single_threaded(self, task_parameters, graph_manager: 'GraphManager', args: argparse.Namespace):
        # Start the training or evaluation
        start_graph(graph_manager=graph_manager, task_parameters=task_parameters)

    def start_multi_threaded(self, graph_manager: 'GraphManager', args: argparse.Namespace):
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
            task_parameters = DistributedTaskParameters(
                framework_type=args.framework,
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
                seed=args.seed+task_index if args.seed is not None else None,  # each worker gets a different seed
                checkpoint_save_secs=args.checkpoint_save_secs,
                checkpoint_restore_dir=args.checkpoint_restore_dir,
                checkpoint_save_dir=args.checkpoint_save_dir,
                export_onnx_graph=args.export_onnx_graph,
                apply_stop_condition=args.apply_stop_condition
            )
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
        if args.evaluation_worker or args.render:
            evaluation_worker = start_distributed_task("worker", args.num_workers, evaluation_worker=True)

        # wait for all workers
        [w.join() for w in workers]
        if args.evaluation_worker:
            evaluation_worker.terminate()


def main():
    launcher = CoachLauncher()
    launcher.launch()


if __name__ == "__main__":
    main()
