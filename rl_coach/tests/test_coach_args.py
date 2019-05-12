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
import pytest
import rl_coach.tests.utils.args_utils as a_utils
import rl_coach.tests.utils.presets_utils as p_utils
from rl_coach.tests.utils.definitions import Definitions as Def


@pytest.mark.functional_test
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

    if flag[0] == "-cp":
        seed = ['--seed', '42']
        seed_flag = a_utils.add_one_flag_value(flag=seed)
        run_cmd.extend(seed_flag)

    run_cmd.extend(test_flag)
    print(str(run_cmd))

    proc = subprocess.Popen(run_cmd, stdout=clres.stdout, stderr=clres.stdout)

    try:
        a_utils.validate_arg_result(flag=test_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)
    except AssertionError:
        # close process once get assert false
        proc.kill()
        assert False

    proc.kill()


@pytest.mark.functional_test
def test_preset_mxnet_framework(preset_for_mxnet_args, clres,
                                start_time=time.time(),
                                time_limit=Def.TimeOuts.test_time_limit):
    """ Test command arguments - the test will check mxnet framework"""

    flag = ['-f', 'mxnet']
    p_valid_params = p_utils.validation_params(preset_for_mxnet_args)

    run_cmd = [
        'python3', 'rl_coach/coach.py',
        '-p', '{}'.format(preset_for_mxnet_args),
        '-e', '{}'.format("ExpName_" + preset_for_mxnet_args),
    ]

    # add flags to run command
    test_flag = a_utils.add_one_flag_value(flag=flag)
    run_cmd.extend(test_flag)

    print(str(run_cmd))

    proc = subprocess.Popen(run_cmd, stdout=clres.stdout, stderr=clres.stdout)

    try:
        a_utils.validate_arg_result(flag=test_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)
    except AssertionError:
        # close process once get assert false
        proc.kill()
        assert False

    proc.kill()


@pytest.mark.functional_test
def test_preset_seed(preset_args_for_seed, clres, start_time=time.time(),
                     time_limit=Def.TimeOuts.test_time_limit):
    """
    Test command arguments - the test will check seed argument with all
    presets
    """

    def close_processes():
        """
        close all processes that still active in the process list
        """
        for i in range(seed_num):
            proc[i].kill()

    proc = []
    seed_num = 2
    flag = ["--seed", str(seed_num)]
    p_valid_params = p_utils.validation_params(preset_args_for_seed)

    run_cmd = [
        'python3', 'rl_coach/coach.py',
        '-p', '{}'.format(preset_args_for_seed),
        '-e', '{}'.format("ExpName_" + preset_args_for_seed),
    ]

    if p_valid_params.trace_test_levels:
        lvl = ['-lvl', '{}'.format(p_valid_params.trace_test_levels[0])]
        run_cmd.extend(lvl)

    # add flags to run command
    test_flag = a_utils.add_one_flag_value(flag=flag)
    run_cmd.extend(test_flag)
    print(str(run_cmd))

    for _ in range(seed_num):
        proc.append(subprocess.Popen(run_cmd, stdout=clres.stdout,
                                     stderr=clres.stdout))

    try:
        a_utils.validate_arg_result(flag=test_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)
    except AssertionError:
        close_processes()
        assert False

    close_processes()


@pytest.mark.functional_test
def test_preset_n_and_ew(preset_args, clres, start_time=time.time(),
                         time_limit=Def.TimeOuts.test_time_limit):
    """
    Test command arguments - check evaluation worker with number of workers
    """

    ew_flag = ['-ew']
    n_flag = ['-n', Def.Flags.enw]
    p_valid_params = p_utils.validation_params(preset_args)

    run_cmd = [
        'python3', 'rl_coach/coach.py',
        '-p', '{}'.format(preset_args),
        '-e', '{}'.format("ExpName_" + preset_args),
    ]

    # add flags to run command
    test_ew_flag = a_utils.add_one_flag_value(flag=ew_flag)
    test_n_flag = a_utils.add_one_flag_value(flag=n_flag)
    run_cmd.extend(test_ew_flag)
    run_cmd.extend(test_n_flag)

    print(str(run_cmd))

    proc = subprocess.Popen(run_cmd, stdout=clres.stdout, stderr=clres.stdout)

    try:
        a_utils.validate_arg_result(flag=test_ew_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)

        a_utils.validate_arg_result(flag=test_n_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)
    except AssertionError:
        # close process once get assert false
        proc.kill()
        assert False

    proc.kill()


@pytest.mark.functional_test
@pytest.mark.xfail(reason="https://github.com/NervanaSystems/coach/issues/257")
def test_preset_n_and_ew_and_onnx(preset_args, clres, start_time=time.time(),
                                  time_limit=Def.TimeOuts.test_time_limit):
    """
    Test command arguments - check evaluation worker, number of workers and
                             onnx.
    """

    ew_flag = ['-ew']
    n_flag = ['-n', Def.Flags.enw]
    onnx_flag = ['-onnx']
    s_flag = ['-s', Def.Flags.css]
    p_valid_params = p_utils.validation_params(preset_args)

    run_cmd = [
        'python3', 'rl_coach/coach.py',
        '-p', '{}'.format(preset_args),
        '-e', '{}'.format("ExpName_" + preset_args),
    ]

    # add flags to run command
    test_ew_flag = a_utils.add_one_flag_value(flag=ew_flag)
    test_n_flag = a_utils.add_one_flag_value(flag=n_flag)
    test_onnx_flag = a_utils.add_one_flag_value(flag=onnx_flag)
    test_s_flag = a_utils.add_one_flag_value(flag=s_flag)

    run_cmd.extend(test_ew_flag)
    run_cmd.extend(test_n_flag)
    run_cmd.extend(test_onnx_flag)
    run_cmd.extend(test_s_flag)

    print(str(run_cmd))

    proc = subprocess.Popen(run_cmd, stdout=clres.stdout, stderr=clres.stdout)

    try:
        # Check csv files has been created
        a_utils.validate_arg_result(flag=test_ew_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)

        # Check csv files created same as the number of the workers
        a_utils.validate_arg_result(flag=test_n_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)

        # Check checkpoint files
        a_utils.validate_arg_result(flag=test_s_flag,
                                    p_valid_params=p_valid_params, clres=clres,
                                    process=proc, start_time=start_time,
                                    timeout=time_limit)

        # TODO: add onnx check; issue found #257

    except AssertionError:
        # close process once get assert false
        proc.kill()
        assert False

    proc.kill()
