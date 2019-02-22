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
import re
import signal
import time

import psutil as psutil

from rl_coach.logger import screen
from rl_coach.tests.utils import test_utils
from rl_coach.tests.utils.definitions import Definitions as Def


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
    for k, v in Def.Flags.cmd_args.items():
        cmd = []
        cmd.append(k)
        if v is not None:
            cmd.append(v)
        assert cmd, Def.Consts.ASSERT_MSG.format("cmd array", str(cmd))
        yield cmd


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

    if Def.Flags.css in flag[1]:
        flag[1] = 30

    elif Def.Flags.crd in flag[1]:
        # TODO: check dir of checkpoint
        flag[1] = os.path.join(Def.Path.experiments)

    elif Def.Flags.et in flag[1]:
        # TODO: add valid value
        flag[1] = ""

    elif Def.Flags.ept in flag[1]:
        # TODO: add valid value
        flag[1] = ""

    elif Def.Flags.cp in flag[1]:
        # TODO: add valid value
        flag[1] = ""

    elif Def.Flags.seed in flag[1]:
        flag[1] = 0

    elif Def.Flags.dccp in flag[1]:
        # TODO: add valid value
        flag[1] = ""

    return flag


def check_files_in_dir(dir_path):
    """
    Check if folder has files
    :param dir_path: |string| folder path
    :return: |Array| return files in folder
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


def find_string_in_logs(log_path, str):
    """
    Find string into the log file
    :param log_path: |string| log path
    :param str: |string| search text
    :return: |bool| true if string found in the log file
    """
    start_time = time.time()
    while time.time() - start_time < Def.TimeOuts.wait_for_files:
        # wait until logs created
        if os.path.exists(log_path):
            break
        time.sleep(1)

    if not os.path.exists(log_path):
        return False

    if str in open(log_path, 'r').read():
        return True
    return False


def get_csv_path(clres):
    """
    Get the csv path with the results - reading csv paths will take some time
    :param clres: object of files that test is creating
    :return: |Array| csv path
    """
    return test_utils.read_csv_paths(test_path=clres.exp_path,
                                     filename_pattern=clres.fn_pattern)


def validate_args_results(flag, clres=None, process=None, start_time=None,
                          timeout=None):
    """
    Validate results of one argument.
    :param flag: flag to check
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
        while time.time() - start_time < timeout:

            if find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.REACHED_REQ_ASC):
                assert True, Def.Consts.ASSERT_MSG. \
                    format(Def.Consts.REACHED_REQ_ASC, "Message Not Found")
                break

    elif flag[0] == "-d" or flag[0] == "--open_dashboard":
        """
        -d, --open_dashboard: Once selected, firefox browser will open to show
                              coach's Dashboard.
        """
        proc_id = None
        start_time = time.time()
        while time.time() - start_time < Def.TimeOuts.wait_for_files:
            for proc in psutil.process_iter():
                if proc.name() == Def.DASHBOARD_PROC:
                    assert proc.name() == Def.DASHBOARD_PROC, \
                        Def.Consts.ASSERT_MSG. format(Def.DASHBOARD_PROC,
                                                      proc.name())
                    proc_id = proc.pid
                    break
            if proc_id:
                break

        if proc_id:
            # kill firefox process
            os.kill(proc_id, signal.SIGKILL)
        else:
            assert False, Def.Consts.ASSERT_MSG.format("Found Firefox process",
                                                       proc_id)

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
        while time.time() - start_time < timeout:

            if find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.COLOR_PREFIX):
                assert True, Def.Consts.ASSERT_MSG. \
                    format(Def.Consts.COLOR_PREFIX, "Color Prefix Not Found")
                break

    elif flag[0] == "--evaluate":
        """
        --evaluate: Once selected, Coach start testing, there is not training.
        """
        tries = 5
        while time.time() - start_time < timeout and tries > 0:

            if find_string_in_logs(log_path=clres.stdout.name,
                                   str=Def.Consts.TRAINING):
                assert False, Def.Consts.ASSERT_MSG.format(
                    "Training Not Found", Def.Consts.TRAINING)
            else:
                time.sleep(1)
                tries -= 1
        assert True, Def.Consts.ASSERT_MSG.format("Training Found",
                                                  Def.Consts.TRAINING)

    elif flag[0] == "--play":
        """
        --play: Once selected alone, warning message should appear, it should
                be with another flag.
        """
        if find_string_in_logs(log_path=clres.stdout.name,
                               str=Def.Consts.PLAY_WARNING):
            assert True, Def.Consts.ASSERT_MSG.format(
                Def.Consts.ONNX_WARNING, "Not found")
