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


from typing import Any, List, Tuple


import numpy as np

from rl_coach.saver import Saver


class ParameterDictSaver(Saver):
    """
    Child class that implements saver for mxnet gluon parameter dictionary
    """
    def __init__(self, name: str, param_dict: gluon.ParameterDict):
        self._name = name
        self._param_dict = param_dict

    @property
    def path(self):
        """
        Relative path for save/load. If two checkpoint objects return the same path, they must be merge-able.
        """
        return self._name

    def save(self, sess: None, save_path: str) -> List[str]:
        """
        Save to save_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param save_path: full path to save checkpoint (typically directory plus self.path plus checkpoint count).
        :return: list of all saved paths
        """
        assert sess is None
        self._param_dict.save(save_path)
        return [save_path]

    def restore(self, sess: Any, restore_path: str):
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: full path to load checkpoint from.
        """
        assert sess is None
        self._param_dict.load(restore_path)

    def merge(self, other: 'Saver'):
        """
        Merge other saver into this saver
        :param other: saver to be merged into self
        """
        if not isinstance(other, ParameterDictSaver):
            raise TypeError('merging only supported with ParameterDictSaver (type:{})'.format(type(other)))
        self._param_dict.update(other._param_dict)

