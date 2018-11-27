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


from typing import Any, List

import tensorflow as tf

from rl_coach.saver import Saver


class GlobalVariableSaver(Saver):
    def __init__(self, name):
        self._names = [name]
        # if graph is finalized, savers must have already already been added. This happens
        # in the case of a MonitoredSession
        self._variables = tf.global_variables()
        
        # target network is never saved or restored directly from checkpoint, so we are removing all its variables from the list
        # the target network would be synched back from the online network in graph_manager.improve(...), at the beginning of the run flow.
        self._variables = [v for v in self._variables if '/target' not in v.name]
        self._saver = tf.train.Saver(self._variables)

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

    def restore(self, sess: Any, restore_path: str):
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: full path to load checkpoint from.
        """
        # We don't use saver.restore() because checkpoint is loaded to online network, but if the checkpoint
        # is from the global network, a namespace mismatch exists and variable name must be modified before loading.
        variables = dict()
        reader = tf.contrib.framework.load_checkpoint(restore_path)
        for var_name, _ in reader.get_variable_to_shape_map().items():
            # if variable was saved using global network, re-map it to online network
            # TODO: Can this be more generic so that `global/` and `online/` are not hardcoded here?
            new_name = var_name.replace('global/', 'online/')
            variables[new_name] = reader.get_tensor(var_name)
        # Assign all variables
        sess.run([v.assign(variables[v.name.split(':')[0]]) for v in self._variables])

    def merge(self, other: 'Saver'):
        """
        Merge other saver into this saver
        :param other: saver to be merged into self
        """
        assert isinstance(other, GlobalVariableSaver)
        self._names.extend(other._names)
        # There is nothing else to do because variables must already be part of the global collection.
