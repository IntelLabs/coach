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
from os import path


def print_progress(averaged_rewards, last_num_episodes, start_time, time_limit,
                   p_valid_params):
    """
    Print progress bar for preset run test
    :param averaged_rewards: average rewards of test
    :param last_num_episodes: last episode number
    :param start_time: start time of test
    :param time_limit: time out of test
    :param p_valid_params: preset validation parameters
    :return:
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


def read_csv_paths(test_path, filename_pattern, read_csv_tries=120):
    """
    Return file path once it found
    :param test_path: test folder path
    :param filename_pattern: csv file pattern
    :param read_csv_tries: number of iterations until file found
    :return: |string| return csv file path
    """
    csv_paths = []
    tries_counter = 0
    while not csv_paths:
        csv_paths = glob.glob(path.join(test_path, '*', filename_pattern))
        if tries_counter > read_csv_tries:
            break
        tries_counter += 1
        time.sleep(1)
    return csv_paths
