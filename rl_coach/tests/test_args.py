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
import subprocess
import time
import rl_coach.tests.utils.args_utils as a_utils
import rl_coach.tests.utils.presets_utils as p_utils
from rl_coach.tests.utils.definitions import Definitions as Def


def test_preset_args(preset_args, flag, clres, start_time=time.time(),
                     time_limit=Def.TimeOuts.test_time_limit):
    """ Test command arguments - the test will check all flags one-by-one."""

    p_valid_params = p_utils.validation_params(preset_args)

    run_cmd = [
        'python3', 'rl_coach/coach.py',
        '-p', '{}'.format(preset_args),
        '-e', '{}'.format("ExpName_" + preset_args),
    ]

    if p_valid_params.reward_test_level:
        lvl = ['-lvl', '{}'.format(p_valid_params.reward_test_level)]
        run_cmd.extend(lvl)

    # add flags to run command
    test_flag = a_utils.add_one_flag_value(flag=flag)
    run_cmd.extend(test_flag)
    print(str(run_cmd))

    # run command
    p = subprocess.Popen(run_cmd, stdout=clres.stdout, stderr=clres.stdout)

    # validate results
    a_utils.validate_args_results(test_flag, clres, p, start_time, time_limit)

    # Close process
    p.kill()
