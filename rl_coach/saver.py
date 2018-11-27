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


"""
Module for abstract base class for checkpoint object and checkpoint collection
"""
from typing import Any, Dict, List


class Saver(object):
    """
    ABC for saver objects that implement saving/restoring to/from path, and merging two savers.
    """
    @property
    def path(self):
        """
        Relative path for save/load. If two saver objects return the same path, they must be merge-able.
        """
        raise NotImplementedError

    def save(self, sess: Any, save_path: str) -> List[str]:
        """
        Save to save_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param save_path: full path to save checkpoint (typically directory plus self.path plus checkpoint count).
        :return: list of all saved paths
        """
        raise NotImplementedError

    def restore(self, sess: Any, restore_path: str) -> None:
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: full path to load checkpoint from.
        """
        raise NotImplementedError

    def merge(self, other: 'Saver') -> None:
        """
        Merge other saver into this saver
        :param other: saver to be merged into self
        """
        raise NotImplementedError


class SaverCollection(object):
    """
    Object for storing a collection of saver objects. It takes care of ensuring uniqueness of saver paths
    and merging savers if they have the same path. For example, if a saver handles saving a generic key/value
    file for all networks in a single file, it can use a more generic path and all savers of all networks would be
    merged into a single saver that saves/restores parameters for all networks.
    NOTE: If two savers have the same path, the respective saver class must support merging them
    into a single saver that saves/restores all merged parameters.
    """
    def __init__(self, saver: Saver = None):
        """
        :param saver: optional initial saver for the collection
        """
        self._saver_dict = dict()  # type: Dict[str, Saver]
        if saver is not None:
            self._saver_dict[saver.path] = saver

    def add(self, saver: Saver):
        """
        Add a new saver to the collection. If saver.path is already in the collection, merge
        the new saver with the existing saver.
        :param saver: new saver to be added to collection
        """
        if saver.path in self._saver_dict:
            self._saver_dict[saver.path].merge(saver)
        else:
            self._saver_dict[saver.path] = saver

    def update(self, other: 'SaverCollection'):
        """
        Merge savers from other collection into self
        :param other: saver collection to update self with.
        """
        for c in other:
            self.add(c)

    def save(self, sess: Any, save_path: str) -> List[str]:
        """
        Call save on all savers in the collection
        :param sess: active session for session-based frameworks (e.g. TF)
        :param save_path: path for saving checkpoints using savers. All saved file paths must
        start with this path in their full path. For example if save_path is '/home/checkpoints/checkpoint-01',
        then saved file paths can be '/home/checkpoints/checkpoint-01.main-network' but not
        '/home/checkpoints/main-network'
        :return: list of all saved paths
        """
        paths = list()
        for saver in self:
            paths.extend(saver.save(sess, self._full_path(save_path, saver)))
        return paths

    def restore(self, sess: Any, restore_path: str) -> None:
        """
        Call restore on all savers in the collection
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: path for restoring checkpoint using savers.
        """
        for saver in self:
            saver.restore(sess, self._full_path(restore_path, saver))

    def __iter__(self):
        """
        Return an iterator for savers in the collection
        :return: saver iterator
        """
        return (v for v in self._saver_dict.values())

    @staticmethod
    def _full_path(path_prefix: str, saver: Saver) -> str:
        """
        Concatenates path of the saver to parent prefix to create full save path
        :param path_prefix: prefix of the path
        :param saver: saver object to get unique path extension from
        :return: full path
        """
        if saver.path == "":
            return path_prefix
        return "{}.{}".format(path_prefix, saver.path)


