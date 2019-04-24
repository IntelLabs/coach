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
"""PyTest configuration."""

import os
import shutil
import sys
import pytest
import rl_coach.tests.utils.args_utils as a_utils
import rl_coach.tests.utils.presets_utils as p_utils
from rl_coach.tests.utils.definitions import Definitions as Def
from os import path


@pytest.fixture(scope="module", params=list(p_utils.collect_presets()))
def preset_name(request):
    """
    Return all preset names
    """
    yield request.param


@pytest.fixture(scope="function", params=list(a_utils.collect_args()))
def flag(request):
    """
    Return flags names in function scope
    """
    yield request.param


@pytest.fixture(scope="module", params=list(a_utils.collect_preset_for_args()))
def preset_args(request):
    """
    Return preset names that can be used for args testing only; working in
    module scope
    """
    yield request.param


@pytest.fixture(scope="module", params=list(a_utils.collect_preset_for_seed()))
def preset_args_for_seed(request):
    """
    Return preset names that can be used for args testing only and for special
    action when using seed argument; working in module scope
    """
    yield request.param


@pytest.fixture(scope="module",
                params=list(a_utils.collect_preset_for_mxnet()))
def preset_for_mxnet_args(request):
    """
    Return preset names that can be used for args testing only; this special
    fixture will be used for mxnet framework only. working in module scope
    """
    yield request.param


@pytest.fixture(scope="function")
def clres(request):
    """
    Create both file csv and log for testing
    :yield: class of both files paths
    """

    class CreateCsvLog:
        """
        Create a test and log paths
        """
        def __init__(self, csv, log, pattern):
            self.exp_path = csv
            self.stdout = open(log, 'w')
            self.fn_pattern = pattern

        @property
        def experiment_path(self):
            return self.exp_path

        @property
        def stdout_path(self):
            return self.stdout

        @experiment_path.setter
        def experiment_path(self, val):
            self.exp_path = val

        @stdout_path.setter
        def stdout_path(self, val):
            self.stdout = open(val, 'w')

    # get preset name from test request params
    idx = 0 if 'preset' in list(request.node.funcargs.items())[0][0] else 1
    p_name = list(request.node.funcargs.items())[idx][1]

    p_valid_params = p_utils.validation_params(p_name)

    sys.path.append('.')
    test_name = 'ExpName_{}'.format(p_name)
    test_path = os.path.join(Def.Path.experiments, test_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)

    # get the stdout for logs results
    log_file_name = 'test_log_{}.txt'.format(p_name)
    fn_pattern = '*.csv' if p_valid_params.num_workers > 1 else 'worker_0*.csv'

    res = CreateCsvLog(test_path, log_file_name, fn_pattern)

    yield res

    # clean files
    if path.exists(res.exp_path):
        shutil.rmtree(res.exp_path)

    if path.exists(res.stdout.name):
        os.remove(res.stdout.name)
