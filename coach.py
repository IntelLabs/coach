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

import sys, inspect, re
import os
import json
import presets
from presets import *
from utils import set_gpu, list_all_classes_in_module
from architectures import *
from environments import *
from agents import *
from utils import *
from logger import screen, logger
import argparse
from subprocess import Popen
import datetime
import presets
import atexit
import sys
import subprocess
from threading import Thread

if len(set(failed_imports)) > 0:
    screen.warning("Warning: failed to import the following packages - {}".format(', '.join(set(failed_imports))))


def set_framework(framework_type):
    # choosing neural network framework
    framework = Frameworks().get(framework_type)
    sess = None
    if framework == Frameworks.TensorFlow:
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        sess = tf.Session(config=config)
    elif framework == Frameworks.Neon:
        import ngraph as ng
        sess = ng.transformers.make_transformer()
    screen.log_title("Using {} framework".format(Frameworks().to_string(framework)))
    return sess


def check_input_and_fill_run_dict(parser):
    args = parser.parse_args()

    # if no arg is given
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    # list available presets
    if args.list:
        presets_lists = list_all_classes_in_module(presets)
        screen.log_title("Available Presets:")
        for preset in presets_lists:
            print(preset)
        sys.exit(0)

    # check inputs
    try:
        # num_workers = int(args.num_workers)
        num_workers = int(re.match("^\d+$", args.num_workers).group(0))
    except ValueError:
        screen.error("Parameter num_workers should be an integer.")

    preset_names = list_all_classes_in_module(presets)
    if args.preset is not None and args.preset not in preset_names:
        screen.error("A non-existing preset was selected. ")

    if args.checkpoint_restore_dir is not None and not os.path.exists(args.checkpoint_restore_dir):
        screen.error("The requested checkpoint folder to load from does not exist. ")

    if args.save_model_sec is not None:
        try:
            args.save_model_sec = int(args.save_model_sec)
        except ValueError:
            screen.error("Parameter save_model_sec should be an integer.")

    if args.preset is None and (args.agent_type is None or args.environment_type is None
                                       or args.exploration_policy_type is None) and not args.play:
        screen.error('When no preset is given for Coach to run, the user is expected to input the desired agent_type,'
                     ' environment_type and exploration_policy_type to assemble a preset. '
                     '\nAt least one of these parameters was not given.')
    elif args.preset is None and args.play and args.environment_type is None:
        screen.error('When no preset is given for Coach to run, and the user requests human control over the environment,'
                     ' the user is expected to input the desired environment_type and level.'
                     '\nAt least one of these parameters was not given.')
    elif args.preset is None and args.play and args.environment_type:
        args.agent_type = 'Human'
        args.exploration_policy_type = 'ExplorationParameters'

    # get experiment name and path
    experiment_name = logger.get_experiment_name(args.experiment_name)
    experiment_path = logger.get_experiment_path(experiment_name)

    if args.play and num_workers > 1:
        screen.warning("Playing the game as a human is only available with a single worker. "
                       "The number of workers will be reduced to 1")
        num_workers = 1

    # fill run_dict
    run_dict = dict()
    run_dict['agent_type'] = args.agent_type
    run_dict['environment_type'] = args.environment_type
    run_dict['exploration_policy_type'] = args.exploration_policy_type
    run_dict['level'] = args.level
    run_dict['preset'] = args.preset
    run_dict['custom_parameter'] = args.custom_parameter
    run_dict['experiment_path'] = experiment_path
    run_dict['framework'] = Frameworks().get(args.framework)
    run_dict['play'] = args.play
    run_dict['evaluate'] = args.evaluate# or args.play

    # multi-threading parameters
    run_dict['num_threads'] = num_workers

    # checkpoints
    run_dict['save_model_sec'] = args.save_model_sec
    run_dict['save_model_dir'] = experiment_path if args.save_model_sec is not None else None
    run_dict['checkpoint_restore_dir'] = args.checkpoint_restore_dir

    # visualization
    run_dict['visualization.dump_gifs'] = args.dump_gifs
    run_dict['visualization.render'] = args.render
    run_dict['visualization.tensorboard'] = args.tensorboard

    return args, run_dict


def run_dict_to_json(_run_dict, task_id=''):
    if task_id != '':
        json_path = os.path.join(_run_dict['experiment_path'], 'run_dict_worker{}.json'.format(task_id))
    else:
        json_path = os.path.join(_run_dict['experiment_path'], 'run_dict.json')

    with open(json_path, 'w') as outfile:
        json.dump(_run_dict, outfile, indent=2)

    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (as configured in presets.py)",
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
                        help="(string) Neural network framework. Available values: tensorflow, neon",
                        default='tensorflow',
                        type=str)
    parser.add_argument('-n', '--num_workers',
                        help="(int) Number of workers for multi-process based agents, e.g. A3C",
                        default='1',
                        type=str)
    parser.add_argument('--play',
                        help="(flag) Play as a human by controlling the game with the keyboard. "
                             "This option will save a replay buffer with the game play.",
                        action='store_true')
    parser.add_argument('--evaluate',
                        help="(flag) Run evaluation only. This is a convenient way to disable "
                             "training in order to evaluate an existing checkpoint.",
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help="(flag) Don't suppress TensorFlow debug prints.",
                        action='store_true')
    parser.add_argument('-s', '--save_model_sec',
                        help="(int) Time in seconds between saving checkpoints of the model.",
                        default=None,
                        type=int)
    parser.add_argument('-crd', '--checkpoint_restore_dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str)
    parser.add_argument('-dg', '--dump_gifs',
                        help="(flag) Enable the gif saving functionality.",
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

    args, run_dict = check_input_and_fill_run_dict(parser)

    # turn TF debug prints off
    if not args.verbose and args.framework.lower() == 'tensorflow':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # dump documentation
    logger.set_dump_dir(run_dict['experiment_path'], add_timestamp=True)
    if not args.no_summary:
        atexit.register(logger.summarize_experiment)
        screen.change_terminal_title(logger.experiment_name)

    # Single-threaded runs
    if run_dict['num_threads'] == 1:
        # set tuning parameters
        json_run_dict_path = run_dict_to_json(run_dict)
        tuning_parameters = json_to_preset(json_run_dict_path)
        tuning_parameters.sess = set_framework(args.framework)

        if args.print_parameters:
            print('tuning_parameters', tuning_parameters)

        # Single-thread runs
        tuning_parameters.task_index = 0
        env_instance = create_environment(tuning_parameters)
        agent = eval(tuning_parameters.agent.type + '(env_instance, tuning_parameters)')

        # Start the training or evaluation
        if tuning_parameters.evaluate:
            agent.evaluate(sys.maxsize, keep_networks_synced=True)  # evaluate forever
        else:
            agent.improve()

    # Multi-threaded runs
    else:
        assert args.framework.lower() == 'tensorflow', "Distributed training works only with TensorFlow"
        os.environ["OMP_NUM_THREADS"]="1"
        # set parameter server and workers addresses
        ps_hosts = "localhost:{}".format(get_open_port())
        worker_hosts = ",".join(["localhost:{}".format(get_open_port()) for i in range(run_dict['num_threads'] + 1)])

        # Make sure to disable GPU so that all the workers will use the CPU
        set_cpu()

        # create a parameter server
        cmd = [
            "python3",
           "./parallel_actor.py",
           "--ps_hosts={}".format(ps_hosts),
           "--worker_hosts={}".format(worker_hosts),
           "--job_name=ps",
        ]
        parameter_server = Popen(cmd)

        screen.log_title("*** Distributed Training ***")
        time.sleep(1)

        # create N training workers and 1 evaluating worker
        workers = []

        for i in range(run_dict['num_threads'] + 1):
            # this is the evaluation worker
            run_dict['task_id'] = i
            if i == run_dict['num_threads']:
                run_dict['evaluate_only'] = True
                run_dict['visualization.render'] = args.render
            else:
                run_dict['evaluate_only'] = False
                run_dict['visualization.render'] = False  # #In a parallel setting, only the evaluation agent renders

            json_run_dict_path = run_dict_to_json(run_dict, i)
            workers_args = ["python3", "./parallel_actor.py",
                            "--ps_hosts={}".format(ps_hosts),
                            "--worker_hosts={}".format(worker_hosts),
                            "--job_name=worker",
                            "--load_json={}".format(json_run_dict_path)]

            p = Popen(workers_args)

            if i != run_dict['num_threads']:
                workers.append(p)
            else:
                evaluation_worker = p

        # wait for all workers
        [w.wait() for w in workers]
        evaluation_worker.kill()
