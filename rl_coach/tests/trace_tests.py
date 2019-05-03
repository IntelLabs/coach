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
import subprocess
import multiprocessing
import sys
import signal
import pandas as pd
import time
from configparser import ConfigParser
from importlib import import_module
from os import path
sys.path.append('.')
from rl_coach.logger import screen


processes = []


def sigint_handler(signum, frame):
    for proc in processes:
        os.killpg(os.getpgid(proc[2].pid), signal.SIGTERM)
    for f in os.listdir('experiments/'):
        if '__test_trace' in f:
            shutil.rmtree(os.path.join('experiments', f))
    for f in os.listdir('.'):
        if 'trace_test_log' in f:
            os.remove(f)
    exit()


signal.signal(signal.SIGINT, sigint_handler)


def read_csv_paths(test_path, filename_pattern, read_csv_tries=100):
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


def run_trace_based_test(preset_name, num_env_steps, level=None):
    test_name = '__test_trace_{}{}'.format(preset_name, '_' + level if level else '').replace(':', '_')
    test_path = os.path.join('./experiments', test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)

    # run the experiment in a separate thread
    screen.log_title("Running test {}{}".format(preset_name, ' - ' + level if level else ''))
    log_file_name = 'trace_test_log_{preset_name}.txt'.format(preset_name=test_name[13:])

    config_file = './tmp.cred'

    cmd = (
        'python3 rl_coach/coach.py '
        '-p {preset_name} ' 
        '-e {test_name} '
        '--seed 42 '
        '-c '
        '-dcp {template}'
        '--no_summary '
        '-cp {custom_param} '
        '{level} '
        '&> {log_file_name} '
    ).format(
        preset_name=preset_name,
        test_name=test_name,
        template=config_file,
        log_file_name=log_file_name,
        level='-lvl ' + level if level else '',
        custom_param='\"improve_steps=EnvironmentSteps({n});'
                     'steps_between_evaluation_periods=EnvironmentSteps({n});'
                     'evaluation_steps=EnvironmentSteps(1);'
                     'heatup_steps=EnvironmentSteps(1024)\"'.format(n=num_env_steps)
    )

    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)

    return test_path, log_file_name, p


def wait_and_check(args, processes, force=False):
    if not force and len(processes) < args.max_threads:
        return None

    test_path = processes[0][0]
    test_name = test_path.split('/')[-1]
    log_file_name = processes[0][1]
    p = processes[0][2]
    p.wait()

    filename_pattern = '*.csv'

    # get the csv with the results
    csv_paths = read_csv_paths(test_path, filename_pattern)

    test_passed = False
    screen.log('Results for {}: '.format(test_name[13:]))
    if not csv_paths:
        screen.error("csv file never found", crash=False)
        if args.verbose:
            screen.error("command exitcode: {}".format(p.returncode), crash=False)
            screen.error(open(log_file_name).read(), crash=False)
    else:
        trace_path = os.path.join('./rl_coach', 'traces', test_name[13:])
        if not os.path.exists(trace_path):
            screen.log('No trace found, creating new trace in: {}'.format(trace_path))
            os.makedirs(trace_path)
            df = pd.read_csv(csv_paths[0])
            df = clean_df(df)
            try:
                df.to_csv(os.path.join(trace_path, 'trace.csv'), index=False)
            except:
                pass
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
    processes.pop(0)
    return test_passed


def generate_config(image, memory_backend, s3_end_point, s3_bucket_name, s3_creds_file, config_file):
    """
    Generate the s3 config file to be used and also the dist-coach-config.template to be used for the test
    It reads the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars and fails if they are not provided.
    """
    # Write s3 creds
    aws_config = ConfigParser({
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
    }, default_section='default')
    with open(s3_creds_file, 'w') as f:
        aws_config.write(f)

    coach_config = ConfigParser({
        'image': image,
        'memory_backend': memory_backend,
        'data_store': 's3',
        's3_end_point': s3_end_point,
        's3_bucket_name': s3_bucket_name,
        's3_creds_file': s3_creds_file
    }, default_section="coach")
    with open(config_file, 'w') as f:
        coach_config.write(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset', '--presets',
                        help="(string) Name of preset(s) to run (comma separated, as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-ip', '--ignore_presets',
                        help="(string) Name of preset(s) to ignore (comma separated, and as configured in presets.py)",
                        default=None,
                        type=str)
    parser.add_argument('-v', '--verbose',
                        help="(flag) display verbose logs in the event of an error",
                        action='store_true')
    parser.add_argument('--stop_after_first_failure',
                        help="(flag) stop executing tests after the first error",
                        action='store_true')
    parser.add_argument('-ow', '--overwrite',
                        help="(flag) overwrite old trace with new ones in trace testing mode",
                        action='store_true')
    parser.add_argument('-prl', '--parallel',
                        help="(flag) run tests in parallel",
                        action='store_true')
    parser.add_argument('-ut', '--update_traces',
                        help="(flag) update traces on repository",
                        action='store_true')
    parser.add_argument('-mt', '--max_threads',
                        help="(int) maximum number of threads to run in parallel",
                        default=multiprocessing.cpu_count()-2,
                        type=int)
    parser.add_argument(
        '-i', '--image', help="(string) Name of the testing image", type=str, default=None
    )
    parser.add_argument(
        '-mb', '--memory_backend', help="(string) Name of the memory backend", type=str, default="redispubsub"
    )
    parser.add_argument(
        '-e', '--endpoint', help="(string) Name of the s3 endpoint", type=str, default='s3.amazonaws.com'
    )
    parser.add_argument(
        '-cr', '--creds_file', help="(string) Path of the s3 creds file", type=str, default='.aws_creds'
    )
    parser.add_argument(
        '-b', '--bucket', help="(string) Name of the bucket for s3", type=str, default=None
    )

    args = parser.parse_args()

    if args.update_traces:
        if not args.bucket:
            print("bucket_name required for s3")
            exit(1)
        if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
            print("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars need to be set")
            exit(1)

        config_file = './tmp.cred'
        generate_config(args.image, args.memory_backend, args.endpoint, args.bucket, args.creds_file, config_file)

    if not args.parallel:
        args.max_threads = 1

    if args.preset is not None:
        presets_lists = args.preset.split(',')
    else:
        presets_lists = [f[:-3] for f in os.listdir(os.path.join('rl_coach', 'presets')) if
                         f[-3:] == '.py' and not f == '__init__.py']

    fail_count = 0
    test_count = 0

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
            num_env_steps = preset_validation_params.trace_max_env_steps
            if preset_validation_params.test_using_a_trace_test:
                if preset_validation_params.trace_test_levels:
                    for level in preset_validation_params.trace_test_levels:
                        test_count += 1
                        test_path, log_file, p = run_trace_based_test(preset_name, num_env_steps, level)
                        processes.append((test_path, log_file, p))
                        test_passed = wait_and_check(args, processes)
                        if test_passed is not None and not test_passed:
                            fail_count += 1
                else:
                    test_count += 1
                    test_path, log_file, p = run_trace_based_test(preset_name, num_env_steps)
                    processes.append((test_path, log_file, p))
                    test_passed = wait_and_check(args, processes)
                    if test_passed is not None and not test_passed:
                        fail_count += 1

    while len(processes) > 0:
        test_passed = wait_and_check(args, processes, force=True)
        if test_passed is not None and not test_passed:
            fail_count += 1

    screen.separator()
    if fail_count == 0:
        screen.success(" Summary: " + str(test_count) + "/" + str(test_count) + " tests passed successfully")
    else:
        screen.error(" Summary: " + str(test_count - fail_count) + "/" + str(test_count) + " tests passed successfully", crash=False)


if __name__ == '__main__':
    os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
    main()
    del os.environ['DISABLE_MUJOCO_RENDERING']
