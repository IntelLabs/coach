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
"""Manage all command arguments."""

import os
import signal
import time
import pandas as pd
import numpy as np
import pytest
from rl_coach.tests.utils.test_utils import get_csv_path, get_files_from_dir, \
    find_string_in_logs
from rl_coach.tests.utils.definitions import Definitions as Def


def collect_preset_for_mxnet():
    """
    Collect presets that relevant for args testing only.
    This used for testing arguments for specific presets that defined in the
    definitions (args_test under Presets).
    :return: preset(s) list
    """
    for pn in Def.Presets.mxnet_args_test:
        assert pn, Def.Consts.ASSERT_MSG.format("Preset name", pn)
        yield pn


def collect_preset_for_args():
    """
    Collect presets that relevant for args testing only.
    This used for testing arguments for specific presets that defined in the
    definitions (args_test under Presets).
    :return: preset(s) list
    """
    for pn in Def.Presets.args_test:
        assert pn, Def.Consts.ASSERT_MSG.format("Preset name", pn)
        yield pn


def collect_preset_for_seed():
    """
    Collect presets that relevant for seed argument testing only.
    This used for testing arguments for specific presets that defined in the
    definitions (args_test under Presets).
    :return: preset(s) list
    """
    for pn in Def.Presets.args_for_seed_test:
        assert pn, Def.Consts.ASSERT_MSG.format("Preset name", pn)
        yield pn


def collect_args():
    """
    Collect args from the cmd args list - on each test iteration, it will
    yield one line (one arg).
    :yield: one arg foe each test iteration
    """
    for i in Def.Flags.cmd_args:
        assert i, Def.Consts.ASSERT_MSG.format("flag list", str(i))
        yield i


def add_one_flag_value(flag):
    """
    Add value to flag format in order to run the python command with arguments.
    :param flag: dict flag
    :return: flag with format
    """
    if not flag or len(flag) == 0:
        return []

    if len(flag) == 1:
        return flag

    if Def.Flags.enw in flag[1]:
        flag[1] = '2'

    elif Def.Flags.css in flag[1]:
        flag[1] = '5'

    elif Def.Flags.fw_ten in flag[1]:
        flag[1] = "tensorflow"

    elif Def.Flags.fw_mx in flag[1]:
        flag[1] = "mxnet"

    elif Def.Flags.cp in flag[1]:
        flag[1] = "heatup_steps=EnvironmentSteps({})".format(Def.Consts.num_hs)

    return flag


def is_reward_reached(csv_path, p_valid_params, start_time, time_limit):
    """
    Check the result of the experiment, by collecting all the Evaluation Reward
    and average should be bigger than the min reward threshold.
    :param csv_path: csv file  (results)
    :param p_valid_params: experiment test params
    :param start_time: start time of the test
    :param time_limit: timeout of the test
    :return: |Bool| true if reached the reward minimum
    """
    win_size = 10
    last_num_episodes = 0
    csv = None
    reward_reached = False

    while csv is None or (csv['Episode #'].values[-1]
           < p_valid_params.max_episodes_to_achieve_reward and
           time.time() - start_time < time_limit):

        csv = pd.read_csv(csv_path)

        if 'Evaluation Reward' not in csv.keys():
            continue

        rewards = csv['Evaluation Reward'].values

        rewards = rewards[~np.isnan(rewards)]
        if len(rewards) >= 1:
            averaged_rewards = np.convolve(rewards, np.ones(
                min(len(rewards), win_size)) / win_size, mode='valid')

        else:
            # May be in heat-up steps
            time.sleep(1)
            continue

        if csv['Episode #'].shape[0] - last_num_episodes <= 0:
            continue

        last_num_episodes = csv['Episode #'].values[-1]

        # check if reward is enough
        if np.any(averaged_rewards >= p_valid_params.min_reward_threshold):
            reward_reached = True
            break
        time.sleep(1)

    return reward_reached


def validate_arg_result(flag, p_valid_params, clres=None, process=None,
                        start_time=None, timeout=Def.TimeOuts.test_time_limit):
    """
    Validate results of one argument.
    :param flag: flag to check
    :param p_valid_params: params test per preset
    :param clres: object of files paths (results of test experiment)
    :param process: process object
    :param start_time: start time of the test
    :param timeout: timeout of the test- fail test once over the timeout
    """

    if flag[0] == "-ns" or flag[0] == "--no-summary":
        """
        --no-summary: Once selected, summary lines shouldn't appear in logs
        """
        # send CTRL+C to close experiment
        process.send_signal(signal.SIGINT)

        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.RESULTS_SORTED), \
            Def.Consts.ASSERT_MSG.format("No Result summary",
                                         Def.Consts.RESULTS_SORTED)

        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.TOTAL_RUNTIME), \
            Def.Consts.ASSERT_MSG.format("No Total runtime summary",
                                         Def.Consts.TOTAL_RUNTIME)

        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.DISCARD_EXP), \
            Def.Consts.ASSERT_MSG.format("No discard message",
                                         Def.Consts.DISCARD_EXP)

    elif flag[0] == "-asc" or flag[0] == "--apply_stop_condition":
        """
        -asc, --apply_stop_condition: Once selected, coach stopped when 
                                      required success rate reached
        """
        assert find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.REACHED_REQ_ASC,
                                   wait_and_find=True), \
            Def.Consts.ASSERT_MSG.format(Def.Consts.REACHED_REQ_ASC,
                                         "Message Not Found")

    elif flag[0] == "--print_networks_summary":
        """
        --print_networks_summary: Once selected, agent summary should appear in
                                  stdout.
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.INPUT_EMBEDDER):
            assert True, Def.Consts.ASSERT_MSG.format(
                Def.Consts.INPUT_EMBEDDER, "Not found")

        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.MIDDLEWARE):
            assert True, Def.Consts.ASSERT_MSG.format(
                Def.Consts.MIDDLEWARE, "Not found")

        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.OUTPUT_HEAD):
            assert True, Def.Consts.ASSERT_MSG.format(
                Def.Consts.OUTPUT_HEAD, "Not found")

    elif flag[0] == "-tb" or flag[0] == "--tensorboard":
        """
        -tb, --tensorboard: Once selected, a new folder should be created in 
                            experiment folder.
        """
        csv_path = get_csv_path(clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        exp_path = os.path.dirname(csv_path[0])
        tensorboard_path = os.path.join(exp_path, Def.Path.tensorboard)

        assert os.path.isdir(tensorboard_path), \
            Def.Consts.ASSERT_MSG.format("tensorboard path", tensorboard_path)

        # check if folder contain files and check extensions
        files = get_files_from_dir(dir_path=tensorboard_path)
        assert any(".tfevents." in file for file in files)

    elif flag[0] == "-onnx" or flag[0] == "--export_onnx_graph":
        """
        -onnx, --export_onnx_graph: Once selected, warning message should 
                                    appear, it should be with another flag.
        """
        assert find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.ONNX_WARNING,
                                   wait_and_find=True), \
            Def.Consts.ASSERT_MSG.format(Def.Consts.ONNX_WARNING, "Not found")

    elif flag[0] == "-dg" or flag[0] == "--dump_gifs":
        """
        -dg, --dump_gifs: Once selected, a new folder should be created in 
                          experiment folder for gifs files.
        """
        pytest.xfail(reason="GUI issue on CI")

        csv_path = get_csv_path(clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        exp_path = os.path.dirname(csv_path[0])
        gifs_path = os.path.join(exp_path, Def.Path.gifs)

        # wait until gif folder were created
        while time.time() - start_time < timeout:
            if os.path.isdir(gifs_path):
                assert os.path.isdir(gifs_path), \
                    Def.Consts.ASSERT_MSG.format("gifs path", gifs_path)
                break

        # check if folder contain files
        get_files_from_dir(dir_path=gifs_path)

    elif flag[0] == "-dm" or flag[0] == "--dump_mp4":
        """
        -dm, --dump_mp4: Once selected, a new folder should be created in 
                         experiment folder for videos files.
        """
        pytest.xfail(reason="GUI issue on CI")

        csv_path = get_csv_path(clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        exp_path = os.path.dirname(csv_path[0])
        videos_path = os.path.join(exp_path, Def.Path.videos)

        # wait until video folder were created
        while time.time() - start_time < timeout:
            if os.path.isdir(videos_path):
                assert os.path.isdir(videos_path), \
                    Def.Consts.ASSERT_MSG.format("videos path", videos_path)
                break

        # check if folder contain files
        get_files_from_dir(dir_path=videos_path)

    elif flag[0] == "--nocolor":
        """
        --nocolor: Once selected, check if color prefix is replacing the actual
                   color; example: '## agent: ...'
        """
        assert find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.COLOR_PREFIX,
                                   wait_and_find=True), \
            Def.Consts.ASSERT_MSG.format(Def.Consts.COLOR_PREFIX,
                                         "Color Prefix Not Found")

    elif flag[0] == "--evaluate":
        """
        --evaluate: Once selected, Coach start testing, there is not training.
        """
        # wait until files created
        get_csv_path(clres=clres)
        time.sleep(15)
        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.TRAINING), \
            Def.Consts.ASSERT_MSG.format("Training Not Found",
                                         Def.Consts.TRAINING)

    elif flag[0] == "--play":
        """
        --play: Once selected alone, an warning message should appear, it 
                should be with another flag.
        """
        assert find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.PLAY_WARNING,
                                   wait_and_find=True), \
            Def.Consts.ASSERT_MSG.format(Def.Consts.PLAY_WARNING, "Not found")

    elif flag[0] == "-et" or flag[0] == "--environment_type":
        """
        -et, --environment_type: Once selected check csv results is created.
        """
        csv_path = get_csv_path(clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

    elif flag[0] == "-s" or flag[0] == "--checkpoint_save_secs":
        """
        -s, --checkpoint_save_secs: Once selected, check if files added to the
                                    experiment path.
        """
        csv_path = get_csv_path(clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        exp_path = os.path.dirname(csv_path[0])
        checkpoint_path = os.path.join(exp_path, Def.Path.checkpoint)

        # wait until video folder were created
        while time.time() - start_time < timeout:
            if os.path.isdir(checkpoint_path):
                assert os.path.isdir(checkpoint_path), \
                    Def.Consts.ASSERT_MSG.format("checkpoint path",
                                                 checkpoint_path)
                break

        # check if folder contain files
        get_files_from_dir(dir_path=checkpoint_path)

    elif flag[0] == "-ew" or flag[0] == "--evaluation_worker":
        """
        -ew, --evaluation_worker: Once selected, check that an evaluation 
                                  worker is created. e.g. by checking that it's
                                  csv file is created.        
        """
        # wait until files created
        csv_path = get_csv_path(clres=clres, extra_tries=10)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

    elif flag[0] == "-cp" or flag[0] == "--custom_parameter":
        """
        -cp, --custom_parameter: Once selected, check that the total steps are
                                 around the given param with +/- gap.
                                 also, check the heat-up param      
        """
        # wait until files created
        csv_path = get_csv_path(clres=clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        # read csv file
        csv = pd.read_csv(csv_path[0])

        # check heat-up value
        results = []
        if csv["In Heatup"].values[-1] == 0:
            results.append(csv["Total steps"].values[-1])
        else:
            while csv["In Heatup"].values[-1] == 1:
                csv = pd.read_csv(csv_path[0])
                last_step = csv["Total steps"].values
                results.append(last_step[-1])
                time.sleep(1)

        # get the first value after heat-up
        time.sleep(3)
        results.append(csv["Total steps"].values[-1])

        assert int(results[-1]) >= Def.Consts.num_hs, \
            Def.Consts.ASSERT_MSG.format("bigger than " +
                                         str(Def.Consts.num_hs), results[-1])

    elif flag[0] == "-f" or flag[0] == "--framework":
        """
        -f, --framework: Once selected, f = tensorflow or mxnet
        """
        # wait until files created
        csv_path = get_csv_path(clres=clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)

        get_reward = is_reward_reached(csv_path=csv_path[0],
                                       p_valid_params=p_valid_params,
                                       start_time=start_time,
                                       time_limit=timeout)

        # check if experiment is working and reached the reward
        assert get_reward, Def.Consts.ASSERT_MSG.format(
            "Doesn't reached the reward", get_reward)

        # check if there is no exception
        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.LOG_ERROR)

        ret_val = process.poll()
        assert ret_val is None, Def.Consts.ASSERT_MSG.format("None", ret_val)

    elif flag[0] == "-crd" or flag[0] == "--checkpoint_restore_dir":

        """
        -crd, --checkpoint_restore_dir: Once selected alone, check that can't
                                        restore checkpoint dir (negative test).
        """
        # wait until files created
        csv_path = get_csv_path(clres=clres)
        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("path not found", csv_path)
        assert find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.NO_CHECKPOINT), \
            Def.Consts.ASSERT_MSG.format(Def.Consts.NO_CHECKPOINT, "Not found")

    elif flag[0] == "--seed":
        """
        --seed: Once selected, check logs of process list if all are the same
                results.
        """
        lst_csv = []
        # wait until files created
        csv_path = get_csv_path(clres=clres, extra_tries=20,
                                num_expected_files=int(flag[1]))

        assert len(csv_path) > 0, \
            Def.Consts.ASSERT_MSG.format("paths are not found", str(csv_path))

        assert int(flag[1]) == len(csv_path), Def.Consts.ASSERT_MSG. \
            format(int(flag[1]), len(csv_path))

        # wait for getting results in csv's
        for i in range(len(csv_path)):

            lines_in_file = pd.read_csv(csv_path[i])
            while len(lines_in_file['Episode #'].values) < 100 and \
                    time.time() - start_time < Def.TimeOuts.test_time_limit:
                lines_in_file = pd.read_csv(csv_path[i])
                time.sleep(1)

            lst_csv.append(pd.read_csv(csv_path[i],
                                       nrows=Def.Consts.N_csv_lines))

        assert len(lst_csv) > 1, Def.Consts.ASSERT_MSG.format("> 1",
                                                              len(lst_csv))

        df1 = lst_csv[0]
        for df in lst_csv[1:]:
            assert list(df1['Training Iter'].values) == list(
                df['Training Iter'].values)

            assert list(df1['ER #Transitions'].values) == list(
                df['ER #Transitions'].values)

            assert list(df1['Total steps'].values) == list(
                df['Total steps'].values)

    elif flag[0] == "-c" or flag[0] == "--use_cpu":
        pass

    elif flag[0] == "-n" or flag[0] == "--num_workers":

        """
        -n, --num_workers: Once selected alone, check that csv created for each
                           worker, and check results.
        """
        # wait until files created
        num_expected_files = int(flag[1])
        csv_path = get_csv_path(clres=clres, extra_tries=20,
                                num_expected_files=num_expected_files)

        assert len(csv_path) >= num_expected_files, \
            Def.Consts.ASSERT_MSG.format(str(num_expected_files),
                                         str(len(csv_path)))

