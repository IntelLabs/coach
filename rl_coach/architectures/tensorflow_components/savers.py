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

import pickle
from typing import Any, List, Dict

import tensorflow as tf
import numpy as np

from rl_coach.saver import Saver


class GlobalVariableSaver(Saver):
    def __init__(self, name):
        self._names = [name]
        # if graph is finalized, savers must have already already been added. This happens
        # in the case of a MonitoredSession
        self._variables = tf.global_variables()

        # target network is never saved or restored directly from checkpoint, so we are removing all its variables from the list
        # the target network would be synched back from the online network in graph_manager.improve(...), at the beginning of the run flow.
        self._variables = [v for v in self._variables if "/target" not in v.name]

        # Using a placeholder to update the variable during restore to avoid memory leak.
        # Ref: https://github.com/tensorflow/tensorflow/issues/4151
        self._variable_placeholders = []
        self._variable_update_ops = []
        for v in self._variables:
            variable_placeholder = tf.placeholder(v.dtype, shape=v.get_shape())
            self._variable_placeholders.append(variable_placeholder)
            self._variable_update_ops.append(v.assign(variable_placeholder))

        self._saver = tf.train.Saver(self._variables, max_to_keep=None)

    @property
    def path(self):
        """
        Relative path for save/load. If two checkpoint objects return the same path, they must be merge-able.
        """
        return ""  # use empty string for global file

    def save(self, sess: None, save_path: str) -> List[str]:
        """
        Save to save_path
        :param sess: active session
        :param save_path: full path to save checkpoint (typically directory plus checkpoint prefix plus self.path)
        :return: list of all saved paths
        """
        save_path = self._saver.save(sess, save_path)
        return [save_path]

    def to_arrays(self, session: Any) -> Dict[str, np.ndarray]:
        """
        Save to dictionary of arrays
        :param sess: active session
        :return: dictionary of arrays
        """
        return {
            k.name.split(":")[0]: v for k, v in zip(self._variables, session.run(self._variables))
        }

    def from_arrays(self, session: Any, tensors: Any):
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param tensors: {name: array}
        """
        # if variable was saved using global network, re-map it to online
        # network
        # TODO: Can this be more generic so that `global/` and `online/` are not
        # hardcoded here?
        if isinstance(tensors, dict):
            tensors = tensors.items()

        variables = {k.replace("global/", "online/"): v for k, v in tensors}

        # Assign all variables using placeholder
        placeholder_dict = {
            ph: variables[v.name.split(":")[0]]
            for ph, v in zip(self._variable_placeholders, self._variables)
        }
        session.run(self._variable_update_ops, placeholder_dict)

    def to_string(self, session: Any) -> str:
        """
        Save to byte string
        :param session: active session
        :return: serialized byte string
        """
        return pickle.dumps(self.to_arrays(session), protocol=-1)

    def from_string(self, session: Any, string: str):
        """
        Restore from byte string
        :param session: active session
        :param string: byte string to restore from
        """
        self.from_arrays(session, pickle.loads(string))

    def _read_tensors(self, restore_path: str):
        """
        Load tensors from a checkpoint
        :param restore_path: full path to load checkpoint from.
        """
        # We don't use saver.restore() because checkpoint is loaded to online
        # network, but if the checkpoint is from the global network, a namespace
        # mismatch exists and variable name must be modified before loading.
        reader = tf.contrib.framework.load_checkpoint(restore_path)
        for var_name, _ in reader.get_variable_to_shape_map().items():
            yield var_name, reader.get_tensor(var_name)

    def restore(self, sess: Any, restore_path: str):
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: full path to load checkpoint from.
        """
        self.from_arrays(sess, self._read_tensors(restore_path))

    def merge(self, other: "Saver"):
        """
        Merge other saver into this saver
        :param other: saver to be merged into self
        """
        assert isinstance(other, GlobalVariableSaver)
        self._names.extend(other._names)
        # There is nothing else to do because variables must already be part of
        # the global collection.
