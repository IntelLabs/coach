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

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
from importlib import import_module
from os import path
sys.path.append('.')
import numpy as np
import pandas as pd
import time

# -*- coding: utf-8 -*-
from rl_coach.logger import screen


def read_csv_paths(test_path, filename_pattern, read_csv_tries=50):
    csv_paths = []
    tries_counter = 0
    while not csv_paths:
        csv_paths = glob.glob(path.join(test_path, '*', filename_pattern))
        if tries_counter > read_csv_tries:
            break
        tries_counter += 1
        time.sleep(1)
    return csv_paths


def clean_df(df):
    if 'Wall-Clock Time' in df.keys():
        df.drop(['Wall-Clock Time'], 1, inplace=True)
    return df


def print_progress(averaged_rewards, last_num_episodes, preset_validation_params, start_time, args):
    percentage = int((100 * last_num_episodes) / preset_validation_params.max_episodes_to_achieve_reward)
    sys.stdout.write("\rReward: ({}/{})".format(round(averaged_rewards[-1], 1),
                                                preset_validation_params.min_reward_threshold))
    sys.stdout.write(' Time (sec): ({}/{})'.format(round(time.time() - start_time, 2), args.time_limit))
    sys.stdout.write(' Episode: ({}/{})'.format(last_num_episodes,
                                                preset_validation_params.max_episodes_to_achieve_reward))
    sys.stdout.write(
        ' {}%|{}{}|  '.format(percentage, '#' * int(percentage / 10), ' ' * (10 - int(percentage / 10))))
    sys.stdout.flush()


def perform_reward_based_tests(args, preset_validation_params, preset_name):
    win_size = 10

    test_name = '__test_reward'
    test_path = os.path.join('./experiments', test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)

    # run the experiment in a separate thread
    screen.log_title("Running test {}".format(preset_name))
    log_file_name = 'test_log_{preset_name}.txt'.format(preset_name=preset_name)
    cmd = (
        'python3 rl_coach/coach.py '
        '-p {preset_name} '
        '-e {test_name} '
        '-n {num_workers} '
        '--seed 0 '
        '-c '
        '{level} '
        '&> {log_file_name} '
    ).format(
        preset_name=preset_name,
        test_name=test_name,
        num_workers=preset_validation_params.num_workers,
        log_file_name=log_file_name,
        level='-lvl ' + preset_validation_params.reward_test_level if preset_validation_params.reward_test_level else ''
    )

    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)

    start_time = time.time()

    reward_str = 'Evaluation Reward'
    if preset_validation_params.num_workers > 1:
        filename_pattern = 'worker_0*.csv'
    else:
        filename_pattern = '*.csv'

    test_passed = False

    # get the csv with the results
    csv_paths = read_csv_paths(test_path, filename_pattern)

    if csv_paths:
        csv_path = csv_paths[0]

        # verify results
        csv = None
        time.sleep(1)
        averaged_rewards = [0]

        last_num_episodes = 0

        if not args.no_progress_bar:
            print_progress(averaged_rewards, last_num_episodes, preset_validation_params, start_time, args)

        while csv is None or (csv['Episode #'].values[
                                  -1] < preset_validation_params.max_episodes_to_achieve_reward and time.time() - start_time < args.time_limit):
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

            if len(rewards) >= 1:
                averaged_rewards = np.convolve(rewards, np.ones(min(len(rewards), win_size)) / win_size, mode='valid')
            else:
                time.sleep(1)
                continue

            if not args.no_progress_bar:
                print_progress(averaged_rewards, last_num_episodes, preset_validation_params, start_time, args)

            if csv['Episode #'].shape[0] - last_num_episodes <= 0:
                continue

            last_num_episodes = csv['Episode #'].values[-1]

            # check if reward is enough
            if np.any(averaged_rewards >= preset_validation_params.min_reward_threshold):
                test_passed = True
                break
            time.sleep(1)

    # kill test and print result
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    screen.log('')
    if test_passed:
        screen.success("Passed successfully")
    else:
        if time.time() - start_time > args.time_limit:
            screen.error("Failed due to exceeding time limit", crash=False)
            if args.verbose:
                screen.error("command exitcode: {}".format(p.returncode), crash=False)
                screen.error(open(log_file_name).read(), crash=False)
        elif csv_paths:
            screen.error("Failed due to insufficient reward", crash=False)
            if args.verbose:
                screen.error("command exitcode: {}".format(p.returncode), crash=False)
                screen.error(open(log_file_name).read(), crash=False)
            screen.error("preset_validation_params.max_episodes_to_achieve_reward: {}".format(
                preset_validation_params.max_episodes_to_achieve_reward), crash=False)
            screen.error("preset_validation_params.min_reward_threshold: {}".format(
                preset_validation_params.min_reward_threshold), crash=False)
            screen.error("averaged_rewards: {}".format(averaged_rewards), crash=False)
            screen.error("episode number: {}".format(csv['Episode #'].values[-1]), crash=False)
        else:
            screen.error("csv file never found", crash=False)
            if args.verbose:
                screen.error("command exitcode: {}".format(p.returncode), crash=False)
                screen.error(open(log_file_name).read(), crash=False)

    shutil.rmtree(test_path)
    os.remove(log_file_name)
    return test_passed


def perform_trace_based_tests(args, preset_name, num_env_steps, level=None):
    test_name = '__test_trace'
    test_path = os.path.join('./experiments', test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)

    # run the experiment in a separate thread
    screen.log_title("Running test {}{}".format(preset_name, ' - ' + level if level else ''))
    log_file_name = 'test_log_{preset_name}.txt'.format(preset_name=preset_name)

    cmd = (
        'python3 rl_coach/coach.py '
        '-p {preset_name} ' 
        '-e {test_name} '
        '--seed 42 '
        '-c '
        '--no_summary '
        '-cp {custom_param} '
        '{level} '
        '&> {log_file_name} '
    ).format(
        preset_name=preset_name,
        test_name=test_name,
        log_file_name=log_file_name,
        level='-lvl ' + level if level else '',
        custom_param='\"improve_steps=EnvironmentSteps({n});'
                     'steps_between_evaluation_periods=EnvironmentSteps({n});'
                     'evaluation_steps=EnvironmentSteps(1);'
                     'heatup_steps=EnvironmentSteps(1024)\"'.format(n=num_env_steps)
    )

    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
    p.wait()

    filename_pattern = '*.csv'

    # get the csv with the results
    csv_paths = read_csv_paths(test_path, filename_pattern)

    test_passed = False
    if not csv_paths:
        screen.error("csv file never found", crash=False)
        if args.verbose:
            screen.error("command exitcode: {}".format(p.returncode), crash=False)
            screen.error(open(log_file_name).read(), crash=False)
    else:
        trace_path = os.path.join('./rl_coach', 'traces', preset_name + '_' + level.replace(':', '_') if level else preset_name, '')
        if not os.path.exists(trace_path):
            screen.log('No trace found, creating new trace in: {}'.format(trace_path))
            os.makedirs(os.path.dirname(trace_path))
            df = pd.read_csv(csv_paths[0])
            df = clean_df(df)
            df.to_csv(os.path.join(trace_path, 'trace.csv'), index=False)
            screen.success("Successfully created new trace.")
            test_passed = True
        else:
            test_df = pd.read_csv(csv_paths[0])
            test_df = clean_df(test_df)
            new_trace_csv_path = os.path.join(trace_path, 'trace_new.csv')
            test_df.to_csv(new_trace_csv_path, index=False)
            test_df = pd.read_csv(new_trace_csv_path)
            trace_csv_path = glob.glob(path.join(trace_path, 'trace.csv'))
            trace_csv_path = trace_csv_path[0]
            trace_df = pd.read_csv(trace_csv_path)
            test_passed = test_df.equals(trace_df)
            if test_passed:
                screen.success("Passed successfully.")
                os.remove(new_trace_csv_path)
                test_passed = True
            else:
                screen.error("Trace test failed.", crash=False)
                if args.overwrite:
                    os.remove(trace_csv_path)
                    os.rename(new_trace_csv_path, trace_csv_path)
                    screen.error("Overwriting old trace.", crash=False)
                else:
                    screen.error("bcompare {} {}".format(trace_csv_path, new_trace_csv_path), crash=False)

    shutil.rmtree(test_path)
    os.remove(log_file_name)
    return test_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace',
                        help="(flag) perform trace based testing",
                        action='store_true')
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-ip', '--ignore_presets',
                        help="(string) Name of a preset(s) to ignore (comma separated, and as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-v', '--verbose',
                        help="(flag) display verbose logs in the event of an error",
                        action='store_true')
    parser.add_argument('--stop_after_first_failure',
                        help="(flag) stop executing tests after the first error",
                        action='store_true')
    parser.add_argument('-tl', '--time_limit',
                        help="time limit for each test in minutes",
                        default=40,  # setting time limit to be so high due to DDPG being very slow - its tests are long
                        type=int)
    parser.add_argument('-np', '--no_progress_bar',
                        help="(flag) Don't print the progress bar (makes jenkins logs more readable)",
                        action='store_true')
    parser.add_argument('-ow', '--overwrite',
                        help="(flag) overwrite old trace with new ones in trace testing mode",
                        action='store_true')

    args = parser.parse_args()
    if args.preset is not None:
        presets_lists = [args.preset]
    else:
        # presets_lists = list_all_classes_in_module(presets)
        presets_lists = [f[:-3] for f in os.listdir(os.path.join('rl_coach', 'presets')) if
                         f[-3:] == '.py' and not f == '__init__.py']

    fail_count = 0
    test_count = 0

    args.time_limit = 60 * args.time_limit

    if args.ignore_presets is not None:
        presets_to_ignore = args.ignore_presets.split(',')
    else:
        presets_to_ignore = []
    for idx, preset_name in enumerate(sorted(presets_lists)):
        if args.stop_after_first_failure and fail_count > 0:
            break
        if preset_name not in presets_to_ignore:
            try:
                preset = import_module('rl_coach.presets.{}'.format(preset_name))
            except:
                screen.error("Failed to load preset <{}>".format(preset_name), crash=False)
                fail_count += 1
                test_count += 1
                continue

            preset_validation_params = preset.graph_manager.preset_validation_params
            if not args.trace and not preset_validation_params.test:
                continue

            if args.trace:
                num_env_steps = preset_validation_params.trace_max_env_steps
                if preset_validation_params.trace_test_levels:
                    for level in preset_validation_params.trace_test_levels:
                        test_count += 1
                        test_passed = perform_trace_based_tests(args, preset_name, num_env_steps, level)
                        if not test_passed:
                            fail_count += 1
                else:
                    test_count += 1
                    test_passed = perform_trace_based_tests(args, preset_name, num_env_steps)
                    if not test_passed:
                        fail_count += 1
            else:
                test_passed = perform_reward_based_tests(args, preset_validation_params, preset_name)
                if not test_passed:
                    fail_count += 1

    screen.separator()
    if fail_count == 0:
        screen.success(" Summary: " + str(test_count) + "/" + str(test_count) + " tests passed successfully")
    else:
        screen.error(" Summary: " + str(test_count - fail_count) + "/" + str(test_count) + " tests passed successfully")


if __name__ == '__main__':
    os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
    main()
    del os.environ['DISABLE_MUJOCO_RENDERING']
