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

# -*- coding: utf-8 -*-
import presets
import numpy as np
import pandas as pd
from os import path
import os
import glob
import shutil
import sys
import time
from logger import screen
from utils import list_all_classes_in_module, threaded_cmd_line_run, killed_processes
import subprocess
import signal
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-ip', '--ignore_presets',
                        help="(string) Name of a preset(s) to ignore (comma separated, and as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-itf', '--ignore_tensorflow',
                        help="(flag) Don't test TensorFlow presets.",
                        action='store_true')
    parser.add_argument('-in', '--ignore_neon',
                        help="(flag) Don't test neon presets.",
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help="(flag) display verbose logs in the event of an error",
                        action='store_true')
    parser.add_argument('-l', '--list_presets',
                        help="(flag) list all the presets that are tested",
                        action='store_true')
    parser.add_argument('--stop_after_first_failure',
                        help="(flag) stop executing tests after the first error",
                        action='store_true')

    args = parser.parse_args()
    if args.preset is not None:
        presets_lists = [args.preset]
    else:
        presets_lists = list_all_classes_in_module(presets)
    win_size = 10
    fail_count = 0
    test_count = 0
    read_csv_tries = 70

    # create a clean experiment directory
    test_name = '__test'
    test_path = os.path.join('./experiments', test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)
    if args.ignore_presets is not None:
        presets_to_ignore = args.ignore_presets.split(',')
    else:
        presets_to_ignore = []

    if args.list_presets:
        for idx, preset_name in enumerate(presets_lists):
            preset = eval('presets.{}()'.format(preset_name))
            if preset.test and preset_name not in presets_to_ignore:
                print(preset_name)
        exit(0)

    for idx, preset_name in enumerate(presets_lists):
        preset = eval('presets.{}()'.format(preset_name))
        if preset.test and preset_name not in presets_to_ignore:
            frameworks = []
            if preset.agent.tensorflow_support and not args.ignore_tensorflow:
                frameworks.append('tensorflow')
            if preset.agent.neon_support and not args.ignore_neon:
                frameworks.append('neon')

            for framework in frameworks:
                if args.stop_after_first_failure and fail_count > 0:
                    break

                test_count += 1

                # run the experiment in a separate thread
                screen.log_title("Running test {} - {}".format(preset_name, framework))
                log_file_name = 'test_log_{preset_name}_{framework}.txt'.format(
                    preset_name=preset_name,
                    framework=framework,
                )
                cmd = (
                    'CUDA_VISIBLE_DEVICES='' python3 coach.py '
                    '-p {preset_name} '
                    '-f {framework} '
                    '-e {test_name} '
                    '-n {num_workers} '
                    '-cp "seed=0" '
                    '&> {log_file_name} '
                ).format(
                    preset_name=preset_name,
                    framework=framework,
                    test_name=test_name,
                    num_workers=preset.test_num_workers,
                    log_file_name=log_file_name,
                )
                p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)

                # get the csv with the results
                csv_path = None
                csv_paths = []

                if preset.test_num_workers > 1:
                    # we have an evaluator
                    reward_str = 'Evaluation Reward'
                    filename_pattern = 'evaluator*.csv'
                else:
                    reward_str = 'Training Reward'
                    filename_pattern = 'worker*.csv'

                initialization_error = False
                test_passed = False

                tries_counter = 0
                while not csv_paths:
                    csv_paths = glob.glob(path.join(test_path, '*', filename_pattern))
                    if tries_counter > read_csv_tries:
                        break
                    tries_counter += 1
                    time.sleep(1)

                if csv_paths:
                    csv_path = csv_paths[0]

                    # verify results
                    csv = None
                    time.sleep(1)
                    averaged_rewards = [0]

                    last_num_episodes = 0
                    while csv is None or csv['Episode #'].values[-1] < preset.test_max_step_threshold:
                        try:
                            csv = pd.read_csv(csv_path)
                        except:
                            # sometimes the csv is being written at the same time we are
                            # trying to read it. no problem -> try again
                            continue

                        if reward_str not in csv.keys():
                            continue

                        rewards = csv[reward_str].values
                        rewards = rewards[~np.isnan(rewards)]

                        if len(rewards) >= win_size:
                            averaged_rewards = np.convolve(rewards, np.ones(win_size) / win_size, mode='valid')
                        else:
                            time.sleep(1)
                            continue

                        # print progress
                        percentage = int((100*last_num_episodes)/preset.test_max_step_threshold)
                        sys.stdout.write("\rReward: ({}/{})".format(round(averaged_rewards[-1], 1), preset.test_min_return_threshold))
                        sys.stdout.write(' Episode: ({}/{})'.format(last_num_episodes, preset.test_max_step_threshold))
                        sys.stdout.write(' {}%|{}{}|  '.format(percentage, '#'*int(percentage/10), ' '*(10-int(percentage/10))))
                        sys.stdout.flush()

                        if csv['Episode #'].shape[0] - last_num_episodes <= 0:
                            continue

                        last_num_episodes = csv['Episode #'].values[-1]

                        # check if reward is enough
                        if np.any(averaged_rewards > preset.test_min_return_threshold):
                            test_passed = True
                            break
                        time.sleep(1)

                # kill test and print result
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                if test_passed:
                    screen.success("Passed successfully")
                else:
                    if csv_paths:
                        screen.error("Failed due to insufficient reward", crash=False)
                        screen.error("preset.test_max_step_threshold: {}".format(preset.test_max_step_threshold), crash=False)
                        screen.error("preset.test_min_return_threshold: {}".format(preset.test_min_return_threshold), crash=False)
                        screen.error("averaged_rewards: {}".format(averaged_rewards), crash=False)
                        screen.error("episode number: {}".format(csv['Episode #'].values[-1]), crash=False)
                    else:
                        screen.error("csv file never found", crash=False)
                        if args.verbose:
                            screen.error("command exitcode: {}".format(p.returncode), crash=False)
                            screen.error(open(log_file_name).read(), crash=False)

                    fail_count += 1
                shutil.rmtree(test_path)


    screen.separator()
    if fail_count == 0:
        screen.success(" Summary: " + str(test_count) + "/" + str(test_count) + " tests passed successfully")
    else:
        screen.error(" Summary: " + str(test_count - fail_count) + "/" + str(test_count) + " tests passed successfully")
