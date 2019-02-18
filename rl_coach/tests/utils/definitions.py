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
"""
Definitions file:

It's main functionality are:
1) housing project constants and enums.
2) housing configuration parameters.
3) housing resource paths.
"""


class Definitions:
    GROUP_NAME = "rl_coach"
    PROCESS_NAME = "coach"
    DASHBOARD_PROC = "firefox"

    class Flags:
        css = "checkpoint_save_secs"
        crd = "checkpoint_restore_dir"
        et = "environment_type"
        ept = "exploration_policy_type"
        cp = "custom_parameter"
        seed = "seed"
        dccp = "distributed_coach_config_path"
        enw = "num_workers"
        fw_ten = "framework_tensorflow"
        fw_mx = "framework_mxnet"
        et = "rl_coach.environments.gym_environment:Atari"

        """
        Arguments that can be tested for python coach command
         ** 1 parameter    = Flag - no need for string or int
         ** 2 parameters   = add value for the selected flag
        """

        cmd_args = [
            ['-ew'],
            ['--play'],
            ['--evaluate'],
            ['-f', fw_ten],
            ['--nocolor'],
            ['-s', css],
            ['-crd', crd],
            ['-dg'],
            ['-dm'],
            ['-cp', cp],
            ['--print_networks_summary'],
            ['-tb'],
            ['-ns'],
            ['-onnx'],
            ['-asc'],
            ['--dump_worker_logs'],
            ['-et', et],
            # '-lvl': '{level}',  # TODO: Add test validation on args_utils
            # '-e': '{}',  # TODO: Add test validation on args_utils
            # '-l': None,  # TODO: Add test validation on args_utils
            # '-c': None,  # TODO: Add test validation using nvidia-smi
            # '-v': None,  # TODO: Add test validation on args_utils
            # '--seed': '{' + seed + '}', # TODO - Add test validation
            # '-dc': None,  # TODO: Add test validation on args_utils
            # '-dcp': '{}'  # TODO: Add test validation on args_utils
            # ['-n', enw],  # Duplicated arg test
            # ['-d'],  # Arg can't be automated - no GUI in the CI
            # '-r': None,  # No automation test
            # '-tfv': None,  # No automation test
            # '-ept': '{' + ept + '}',  # No automation test - not supported
        ]
        # TODO: 1- add params that still not tested (from above list)
        #       2- remove irrelevant args
        cmd_args_combination = [
            ['-ew'],
            ['-f', fw_ten],
            ['--evaluate'],
            ['-s', css],
            ['-dg'],
            ['-dm'],
            ['-cp', cp],
            ['-tb'],
            ['-ns'],
            ['-ns'],
            ['-onnx'],
            ['-asc'],
        ]

    class Presets:
        # Preset list for testing the flags/ arguments of python coach command
        args_test = [
            "CartPole_A3C",
        ]

    class PresetsForMXNet:
        # Preset list for testing the flags/ arguments of python coach command
        args_test = [
            "Doom_Basic_DQN",
        ]

    class Path:
        experiments = "./experiments"
        tensorboard = "tensorboard"
        gifs = "gifs"
        videos = "videos"
        checkpoint = "checkpoint"

    class Consts:
        ASSERT_MSG = "Expected: {}, Actual: {}."
        RESULTS_SORTED = "Results stored at:"
        TOTAL_RUNTIME = "Total runtime:"
        DISCARD_EXP = "Do you want to discard the experiment results"
        REACHED_REQ_ASC = "Reached required success rate. Exiting."
        INPUT_EMBEDDER = "Input Embedder:"
        MIDDLEWARE = "Middleware:"
        OUTPUT_HEAD = "Output Head:"
        ONNX_WARNING = "Exporting ONNX graphs requires setting the " \
                       "--checkpoint_save_secs flag. The --export_onnx_graph" \
                       " will have no effect."
        COLOR_PREFIX = "## agent: Starting evaluation phase"
        TRAINING = "Training - "
        PLAY_WARNING = "Both the --preset and the --play flags were set. " \
                       "These flags can not be used together. For human " \
                       "control, please use the --play flag together with " \
                       "the environment type flag (-et)"
        NO_CHECKPOINT = "No checkpoint to restore in:"
        LOG_ERROR = "KeyError:"

        num_is = 2000
        num_sbep = 500
        num_hs = 200
        num_es = 1

        f_comb = 2  # number of flags in cmd for creating flags combinations

    class TimeOuts:
        test_time_limit = 60 * 60
        wait_for_files = 20
        wait_for_csv = 240
        test_run = 15
