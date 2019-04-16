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
"""Manage all preset"""

import os
import pytest
from importlib import import_module
from rl_coach.tests.utils.definitions import Definitions as Def


def import_preset(preset_name):
    """
    Import preset name module from presets directory
    :param preset_name: preset name
    :return: imported module
    """
    try:
        module = import_module('{}.presets.{}'
                               .format(Def.GROUP_NAME, preset_name))
    except:
        pytest.skip("Can't import module: {}".format(preset_name))

    return module


def validation_params(preset_name):
    """
    Validate parameters based on preset name
    :param preset_name: preset name
    :return: |bool| true if preset has params
    """
    return import_preset(preset_name).graph_manager.preset_validation_params


def all_presets():
    """
    Get all preset from preset directory
    :return: |Array| preset list
    """
    return [
        f[:-3] for f in os.listdir(os.path.join(Def.GROUP_NAME, 'presets'))
        if f[-3:] == '.py' and not f == '__init__.py'
    ]


def importable(preset_name):
    """
    Try to import preset name
    :param preset_name: |name| preset name
    :return: |bool| true if possible to import preset
    """
    try:
        import_preset(preset_name)
        return True
    except BaseException:
        return False


def has_test_parameters(preset_name):
    """
    Check if preset has parameters
    :param preset_name: |string| preset name
    :return: |bool| true: if preset have parameters
    """
    return bool(validation_params(preset_name).test)


def collect_presets():
    """
    Collect all presets in presets directory
    :yield: preset name
    """
    for preset_name in all_presets():
        # if it isn't importable, still include it so we can fail the test
        if not importable(preset_name):
            yield preset_name
        # otherwise, make sure it has test parameters before including it
        elif has_test_parameters(preset_name):
            yield preset_name
