#
# Copyright (c) 2019 Intel Corporation
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
"""Common functionality shared across tests."""

import glob
import sys
import time
import os
from os import path
from rl_coach.tests.utils.definitions import Definitions as Def


def print_progress(averaged_rewards, last_num_episodes, start_time, time_limit,
                   p_valid_params):
    """
    Print progress bar for preset run test
    :param averaged_rewards: average rewards of test
    :param last_num_episodes: last episode number
    :param start_time: start time of test
    :param time_limit: time out of test
    :param p_valid_params: preset validation parameters
    """
    max_episodes_to_archive = p_valid_params.max_episodes_to_achieve_reward
    min_reward = p_valid_params.min_reward_threshold
    avg_reward = round(averaged_rewards[-1], 1)
    percentage = int((100 * last_num_episodes) / max_episodes_to_archive)
    cur_time = round(time.time() - start_time, 2)

    sys.stdout.write("\rReward: ({}/{})".format(avg_reward, min_reward))
    sys.stdout.write(' Time (sec): ({}/{})'.format(cur_time, time_limit))
    sys.stdout.write(' Episode: ({}/{})'.format(last_num_episodes,
                                                max_episodes_to_archive))
    sys.stdout.write(' {}%|{}{}|  '
                     .format(percentage, '#' * int(percentage / 10),
                             ' ' * (10 - int(percentage / 10))))

    sys.stdout.flush()


def read_csv_paths(test_path, filename_pattern, read_csv_tries=120,
                   extra_tries=0, num_expected_files=None):
    """
    Return file path once it found
    :param test_path: |string| test folder path
    :param filename_pattern: |string| csv file pattern
    :param read_csv_tries: |int| number of iterations until file found
    :param extra_tries: |int| add number of extra tries to check after getting
                        all the paths.
    :param num_expected_files: find all expected file in experiment folder.
    :return: |string| return csv file path
    """
    csv_paths = []
    tries_counter = 0

    if isinstance(extra_tries, int) and extra_tries >= 0:
        read_csv_tries += extra_tries

    while tries_counter < read_csv_tries:
        csv_paths = glob.glob(path.join(test_path, '*', filename_pattern))

        if num_expected_files:
            if num_expected_files == len(csv_paths):
                break
            else:
                time.sleep(1)
                tries_counter += 1
                continue
        elif csv_paths:
            break

        time.sleep(1)
        tries_counter += 1

    return csv_paths


def get_files_from_dir(dir_path):
    """
    Check if folder has files
    :param dir_path: |string| folder path
    :return: |list| return files in folder
    """
    start_time = time.time()
    entities = []
    while time.time() - start_time < Def.TimeOuts.wait_for_files:
        # wait until logs created
        if os.path.exists(dir_path):
            entities = os.listdir(dir_path)
            if len(entities) > 0:
                break
        time.sleep(1)

    assert len(entities) > 0, \
        Def.Consts.ASSERT_MSG.format("num files > 0", len(entities))
    return entities


def find_string_in_logs(log_path, str, timeout=Def.TimeOuts.wait_for_files,
                        wait_and_find=False):
    """
    Find string into the log file
    :param log_path: |string| log path
    :param str: |string| search text
    :param timeout: |int| timeout for searching on file
    :param wait_and_find: |bool| true if need to wait until reaching timeout
    :return: |bool| true if string found in the log file
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        # wait until logs created
        if os.path.exists(log_path):
            break
        time.sleep(1)

    if not os.path.exists(log_path):
        return False

    while time.time() - start_time < Def.TimeOuts.test_time_limit:
        with open(log_path, 'r') as fr:
            if str in fr.read():
                return True
            fr.close()

        if not wait_and_find:
            break

    return False


def get_csv_path(clres, tries_for_csv=Def.TimeOuts.wait_for_csv,
                 extra_tries=0, num_expected_files=None):
    """
    Get the csv path with the results - reading csv paths will take some time
    :param clres: object of files that test is creating
    :param tries_for_csv: timeout of tires until getting all csv files
    :param extra_tries: add number of extra tries to check after getting all
                        the paths.
    :param num_expected_files: find all expected file in experiment folder.
    :return: |list| csv path
    """
    return read_csv_paths(test_path=clres.exp_path,
                          filename_pattern=clres.fn_pattern,
                          read_csv_tries=tries_for_csv,
                          extra_tries=extra_tries,
                          num_expected_files=num_expected_files)

