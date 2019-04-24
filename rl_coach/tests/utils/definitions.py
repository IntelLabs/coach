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
        cp = "custom_parameter"
        seed = "seed"
        dccp = "distributed_coach_config_path"
        enw = "num_workers"
        fw_ten = "framework_tensorflow"
        fw_mx = "framework_mxnet"
        # et = "rl_coach.environments.gym_environment:Atari" TODO

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
            # ['-crd', crd], # Tested in checkpoint test
            ['-dg'],
            ['-dm'],
            ['-cp', cp],
            ['--print_networks_summary'],
            ['-tb'],
            ['-ns'],
            ['-onnx'],
            ['-asc'],
            ['--dump_worker_logs'],
            # ['-et', et],
            # '-lvl': '{level}',  # TODO: Add test validation on args_utils
            # '-e': '{}',  # TODO: Add test validation on args_utils
            # '-l': None,  # TODO: Add test validation on args_utils
            # '-c': None,  # TODO: Add test validation using nvidia-smi
            # '-v': None,  # TODO: Add test validation on args_utils
            # '--seed': '{' + seed + '}', # DONE - new test added
            # '-dc': None,  # TODO: Add test validation on args_utils
            # '-dcp': '{}'  # TODO: Add test validation on args_utils
            # ['-n', enw],  # Duplicated arg test
            # ['-d'],  # Arg can't be automated - no GUI in the CI
            # '-r': None,  # No automation test
            # '-tfv': None,  # No automation test
            # '-ept': '{' + ept + '}',  # No automation test - not supported
        ]

    class Presets:
        # Preset list for testing the flags/ arguments of python coach command
        args_test = [
            "CartPole_A3C",
        ]

        # Preset list for mxnet framework testing
        mxnet_args_test = [
            "CartPole_DQN"
        ]

        # Preset for testing seed argument
        args_for_seed_test = [
            "Atari_DQN",
            "Doom_Basic_DQN",
            "BitFlip_DQN",
            "CartPole_DQN",
            "CARLA_Dueling_DDQN",
            "ControlSuite_DDPG",
            "ExplorationChain_Dueling_DDQN",
            "Fetch_DDPG_HER_baselines",
            "Mujoco_ClippedPPO",
            "Starcraft_CollectMinerals_Dueling_DDQN",
        ]

    class Path:
        experiments = "./experiments"
        tensorboard = "tensorboard"
        test_dir = "test_dir"
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

        num_hs = 200  # heat-up steps (used for agent custom parameters)

        f_comb = 2  # number of flags in cmd for creating flags combinations

        N_csv_lines = 100  # number of lines to validate on csv file

    class TimeOuts:
        test_time_limit = 60 * 60
        wait_for_files = 20
        wait_for_csv = 240
        test_run = 60
