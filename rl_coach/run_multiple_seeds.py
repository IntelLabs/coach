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

import sys
sys.path.append('.')
from subprocess import Popen
import argparse
from rl_coach.utils import set_gpu, force_list

"""
This script makes it easier to run multiple instances of a given preset.
Each instance uses a different seed, and optionally, multiple environment levels can be configured as well.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) The preset to run",
                        default=None,
                        type=str)
    parser.add_argument('-s', '--seeds',
                        help="(int) Number of seeds to run",
                        default=5,
                        type=int)
    parser.add_argument('-lvl', '--level',
                        help="(string) Environment level to use. This can be defined as a comma separated list.",
                        default=None,
                        type=str)
    parser.add_argument('-g', '--gpu',
                        help="(int) The gpu to use. This can be defined as a comma separated list. For example,"
                             " 0,1 will use both gpu's 0 and 1, by switching between them for each run instance",
                        default='0',
                        type=str)
    parser.add_argument('-n', '--num_workers',
                        help="(int) The number of workers to use for each run",
                        default=1,
                        type=int)
    parser.add_argument('-d', '--dir_prefix',
                        help="(str) A prefix for the directory name. If not given, the directory name will match "
                             "the preset name, followed by the environment level",
                        default='',
                        type=str)
    parser.add_argument('-sd', '--level_as_sub_dir',
                        help="(flag) Store each level in it's own sub directory where the root directory name matches "
                             "the preset name",
                        action='store_true')
    parser.add_argument('-ew', '--evaluation_worker',
                        help="(flag) Start an additional worker that will only do evaluation",
                        action='store_true')
    parser.add_argument('-c', '--use_cpu',
                        help="(flag) Use the cpu instead of the gpu",
                        action='store_true')
    args = parser.parse_args()

    # dir_prefix = "benchmark_"
    # preset = 'Mujoco_DDPG'  # 'Mujoco_DDPG'
    # levels = ["inverted_pendulum"]
    # num_seeds = 5
    # num_workers = 1
    # gpu = [0, 1]
    #

    # if no arg is given
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    dir_prefix = args.dir_prefix
    preset = args.preset
    levels = args.level.split(',') if args.level is not None else [None]
    num_seeds = args.seeds
    num_workers = args.num_workers
    gpu = [int(gpu) for gpu in args.gpu.split(',')]
    level_as_sub_dir = args.level_as_sub_dir

    processes = []
    gpu_list = force_list(gpu)
    curr_gpu_idx = 0
    for level in levels:
        if dir_prefix != "":
            dir_prefix += "_"
        for seed in range(num_seeds):
            # select the next gpu for this run
            set_gpu(gpu_list[curr_gpu_idx])

            command = ['python3', 'rl_coach/coach.py', '-ns', '-p', '{}'.format(preset),
                        '--seed', '{}'.format(seed), '-n', '{}'.format(num_workers)]
            if args.use_cpu:
                command.append("-c")
            if args.evaluation_worker:
                command.append("-ew")
            if level is not None:
                command.extend(['-lvl', '{}'.format(level)])
                if level_as_sub_dir:
                    separator = "/"
                else:
                    separator = "_"
                command.extend(['-e', '{dir_prefix}{preset}{separator}{level}_{num_workers}_workers'.format(
                    dir_prefix=dir_prefix, preset=preset, level=level, separator=separator, num_workers=args.num_workers)])
            else:
                command.extend(['-e', '{dir_prefix}{preset}_{num_workers}_workers'.format(
                    dir_prefix=dir_prefix, preset=preset, num_workers=args.num_workers)])
            print(command)

            p = Popen(command)
            processes.append(p)

            # for each run, select the next gpu from the available gpus
            curr_gpu_idx = (curr_gpu_idx + 1) % len(gpu_list)

    for p in processes:
        p.wait()
