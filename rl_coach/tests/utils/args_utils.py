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
from itertools import combinations
from rl_coach.tests.utils import test_utils
from rl_coach.tests.utils.definitions import Definitions as Def


def collect_preset_for_mxnet():
    """
    Collect presets that relevant for args testing only.
    This used for testing arguments for specific presets that defined in the
    definitions (args_test under Presets).
    :return: preset(s) list
    """
    for pn in Def.PresetsForMXNet.args_test:
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


def collect_args():
    """
    Collect args from the cmd args list - on each test iteration, it will
    yield one line (one arg).
    :yield: one arg foe each test iteration
    """
    for i in Def.Flags.cmd_args:
        assert i, Def.Consts.ASSERT_MSG.format("flag list", str(i))
        yield i


def collect_args_comb():
    """
    Collect args from the cmd args list - in each test iteration, it will,
    yield a bunch of args (depends on the Consts.f_comb value).
    :yield: bunch of flags
    """
    comb = combinations(Def.Flags.cmd_args_combination, Def.Consts.f_comb)
    for i in list(comb):
        assert i, Def.Consts.ASSERT_MSG.format("flag list", str(i))
        yield i


def add_combination_flags(comb_flags):
    """
    Extend flag list to one list
    :param comb_flags: list of flags with values
    :return: |list| list of all flags together
    """
    assert len(comb_flags) == Def.Consts.f_comb, Def.Consts.ASSERT_MSG.format(
        "combination flag should be equal to const" + Def.Consts.f_comb,
        len(comb_flags))

    flags_arr = []
    for flag in comb_flags:
        flags_arr.extend(add_one_flag_value(flag))

    return flags_arr


def add_one_flag_value(flag):
    """
    Add value to flag format in order to run the python command with arguments.
    :param flag: dict flag
    :return: flag with format
    """
    if not flag or len(flag) > 2 or len(flag) == 0:
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
        flag[1] = "evaluation_steps=EnvironmentSteps({});" \
                  "heatup_steps=EnvironmentSteps({});" \
                  "steps_between_evaluation_periods=EnvironmentSteps({});" \
                  "improve_steps=EnvironmentSteps({})" \
            .format(Def.Consts.num_es, Def.Consts.num_hs, Def.Consts.num_sbep,
                    Def.Consts.num_is)

    elif Def.Flags.crd in flag[1]:
        flag[1] = Def.Path.experiments

    return flag


def check_files_in_dir(dir_path):
    """
    Check if folder has files
    :param dir_path: |string| folder path
    :return: |list| return files in folder
    """
    start_time = time.time()
    entities = None
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


def find_string_in_logs(log_path, str, timeout=Def.TimeOuts.wait_for_files):
    """
    Find string into the log file
    :param log_path: |string| log path
    :param str: |string| search text
    :param timeout: |int| timeout for searching on file
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

    with open(log_path, 'r') as fr:
        if str in fr.read():
            return True
    return False


def get_csv_path(clres, tires_for_csv=Def.TimeOuts.wait_for_csv):
    """
    Get the csv path with the results - reading csv paths will take some time
    :param clres: object of files that test is creating
    :param tires_for_csv: timeout of tires until getting all csv files
    :return: |list| csv path
    """
    return test_utils.read_csv_paths(test_path=clres.exp_path,
                                     filename_pattern=clres.fn_pattern,
                                     read_csv_tries=tires_for_csv)


def is_reward_reached(csv_path, p_valid_params, start_time, time_limit):
    """
    Check the result of the experiment, by collecting all the Evaluation Reward
    and aviarage should be bigger than the min reward threshold.
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
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.REACHED_REQ_ASC):
            assert True, Def.Consts.ASSERT_MSG. \
                format(Def.Consts.REACHED_REQ_ASC, "Message Not Found")

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

        # check if folder contain files
        check_files_in_dir(dir_path=tensorboard_path)

    elif flag[0] == "-onnx" or flag[0] == "--export_onnx_graph":
        """
        -onnx, --export_onnx_graph: Once selected, warning message should 
                                    appear, it should be with another flag.
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.ONNX_WARNING):
            assert True, Def.Consts.ASSERT_MSG.format(
                Def.Consts.ONNX_WARNING, "Not found")

    elif flag[0] == "-dg" or flag[0] == "--dump_gifs":
        """
        -dg, --dump_gifs: Once selected, a new folder should be created in 
                          experiment folder for gifs files.
        """
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
        check_files_in_dir(dir_path=gifs_path)
        # TODO: check if play window is opened

    elif flag[0] == "-dm" or flag[0] == "--dump_mp4":
        """
        -dm, --dump_mp4: Once selected, a new folder should be created in 
                         experiment folder for videos files.
        """
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
        check_files_in_dir(dir_path=videos_path)
        # TODO: check if play window is opened

    elif flag[0] == "--nocolor":
        """
        --nocolor: Once selected, check if color prefix is replacing the actual
                   color; example: '## agent: ...'
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.COLOR_PREFIX):
            assert True, Def.Consts.ASSERT_MSG. \
                format(Def.Consts.COLOR_PREFIX, "Color Prefix Not Found")

    elif flag[0] == "--evaluate":
        """
        --evaluate: Once selected, Coach start testing, there is not training.
        """
        # wait until files created
        get_csv_path(clres=clres)
        assert not find_string_in_logs(log_path=clres.stdout.name,
                                       str=Def.Consts.TRAINING), \
            Def.Consts.ASSERT_MSG.format("Training Not Found",
                                         Def.Consts.TRAINING)

    elif flag[0] == "--play":
        """
        --play: Once selected alone, an warning message should appear, it 
                should be with another flag.
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.PLAY_WARNING):
            assert True, Def.Consts.ASSERT_MSG.format(
                  Def.Consts.PLAY_WARNING, "Not found")

    elif flag[0] == "-et" or flag[0] == "--environment_type":
        """
        -et, --environment_type: Once selected alone, an warning message should
                appear, it should be with another flag.
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.PLAY_WARNING):
            assert True, Def.Consts.ASSERT_MSG.format(
                  Def.Consts.PLAY_WARNING, "Not found")

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
        check_files_in_dir(dir_path=checkpoint_path)

    elif flag[0] == "-ew" or flag[0] == "--evaluation_worker":
        """
        -ew, --evaluation_worker: Once selected, check that an evaluation 
                                  worker is created. e.g. by checking that it's
                                  csv file is created.        
        """
        # wait until files created
        csv_path = get_csv_path(clres=clres)
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

        # wait until finish the experiment
        find_string_in_logs(log_path=clres.stdout.name,
                            str=Def.Consts.RESULTS_SORTED, timeout=240)

        # read csv file
        csv = pd.read_csv(csv_path[0])

        # check total step value
        totalstep = csv['Total steps'].values[-1]
        assert (totalstep <= (Def.Consts.num_is + 300)
                and totalstep >= (Def.Consts.num_is - 300)), \
            Def.Consts.ASSERT_MSG.format("should be around" +
                                         str(Def.Consts.num_is), totalstep)

        # check heatup value
        last_row_heatup = len(np.nonzero(csv["In Heatup"].values)[0])
        total_step_heatup = csv['Total steps'].values[last_row_heatup - 1]
        assert (total_step_heatup <= (Def.Consts.num_hs + 30) and
                total_step_heatup >= (Def.Consts.num_hs - 30)), \
            Def.Consts.ASSERT_MSG.format("should be around" +
                                         str(Def.Consts.num_hs),
                                         total_step_heatup)

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
