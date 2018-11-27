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
Module providing helper classes and functions for reading/writing checkpoint state
"""

import os
import re
from typing import List, Union, Tuple


class SingleCheckpoint(object):
    """
    Helper class for storing checkpoint name and number
    """
    def __init__(self, num: int, name: str):
        """
        :param num: checkpoint number
        :param name: checkpoint name (i.e. the prefix for all checkpoint files)
        """
        self._num = num
        self._name = name

    @property
    def num(self) -> int:
        return self._num

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return self._name

    def __repr__(self):
        return str(self)

    def __eq__(self, other: 'SingleCheckpoint'):
        if not isinstance(other, SingleCheckpoint):
            return False
        return self._name == other._name and self._num == other._num

    def __ne__(self, other):
        return not self.__eq__(other)


class CheckpointState(object):
    """
    Helper class for checkpoint directory information. It replicates
    the CheckpointState protobuf class in tensorflow with addition of
    two new functions: last_checkpoint() and all_checkpoints()
    """
    def __init__(self, checkpoints: List[SingleCheckpoint], checkpoint_dir: str):
        """
        :param checkpoints: sorted list of checkpoints from oldest to newest. checkpoint[-1] is
            considered to be the most recent checkpoint.
        :param checkpoint_dir: checkpoint directory which is added to the paths
        """
        self._checkpoints = checkpoints
        self._checkpoin_dir = checkpoint_dir

    @property
    def all_checkpoints(self) -> List[SingleCheckpoint]:
        """
        :return: list of all checkpoints
        """
        return self._checkpoints

    @property
    def last_checkpoint(self) -> SingleCheckpoint:
        """
        :return: the most recent checkpoint
        """
        return self._checkpoints[-1]

    @property
    def all_model_checkpoint_paths(self) -> List[str]:
        """
        TF compatible function call to get all checkpoints
        :return: list of all available model checkpoint paths
        """
        return [os.path.join(self._checkpoin_dir, c.name) for c in self._checkpoints]

    @property
    def model_checkpoint_path(self) -> str:
        """
        TF compatible call to get most recent checkpoint
        :return: path of the most recent model checkpoint
        """
        return os.path.join(self._checkpoin_dir, self._checkpoints[-1].name)

    def __str__(self):
        out_str = 'model_checkpoint_path: {}\n'.format(self.model_checkpoint_path)
        for c in self.all_model_checkpoint_paths:
            out_str += 'all_model_checkpoint_paths: {}\n'.format(c)
        return out_str

    def __repr__(self):
        return str(self._checkpoints)


class CheckpointStateFile(object):
    """
    Helper class for reading from and writing to the checkpoint state file
    """
    checkpoint_state_filename = '.coach_checkpoint'

    def __init__(self, checkpoint_dir: str):
        self._checkpoint_state_path = os.path.join(checkpoint_dir, self.checkpoint_state_filename)

    def exists(self) -> bool:
        """
        :return: True if checkpoint state file exists, false otherwise
        """
        return os.path.exists(self._checkpoint_state_path)

    def read(self) -> Union[None, SingleCheckpoint]:
        """
        Read checkpoint state file and interpret its content
        :return:
        """
        if not self.exists():
            return None
        with open(self._checkpoint_state_path, 'r') as fd:
            return CheckpointFilenameParser().parse(fd.read(256))

    def write(self, data: SingleCheckpoint) -> None:
        """
        Writes data to checkpoint state file
        :param data: string data
        """
        with open(self._checkpoint_state_path, 'w') as fd:
            fd.write(data.name)

    @property
    def filename(self) -> str:
        return self.checkpoint_state_filename

    @property
    def path(self) -> str:
        return self._checkpoint_state_path


class CheckpointStateReader(object):
    """
    Class for scanning checkpoint directory and updating the checkpoint state
    """
    def __init__(self, checkpoint_dir: str, checkpoint_state_optional: bool=True):
        """
        :param checkpoint_dir: path to checkpoint directory
        :param checkpoint_state_optional: If True, checkpoint state file is optional and if not found,
            directory is scanned to find the latest checkpoint. Default is True for backward compatibility
        """
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_state_file = CheckpointStateFile(self._checkpoint_dir)
        self._checkpoint_state_optional = checkpoint_state_optional

    def get_latest(self) -> SingleCheckpoint:
        """
        Tries to read the checkpoint state file. If that fails, discovers latest by reading the entire directory.
        :return: checkpoint object representing the latest checkpoint
        """
        latest = self._checkpoint_state_file.read()
        if latest is None and self._checkpoint_state_optional:
            all_checkpoints = _filter_checkpoint_files(os.listdir(self._checkpoint_dir))
            if len(all_checkpoints) > 0:
                latest = all_checkpoints[-1]
        return latest

    def get_all(self) -> List[SingleCheckpoint]:
        """
        Reads both the checkpoint state file as well as contents of the directory and merges them into one list.
        :return: list of checkpoint objects
        """
        # discover all checkpoint files in directory if requested or if a valid checkpoint-state file doesn't exist
        all_checkpoints = _filter_checkpoint_files(os.listdir(self._checkpoint_dir))
        last_checkpoint = self._checkpoint_state_file.read()
        if last_checkpoint is not None:
            # remove excess checkpoints: higher checkpoint number, but not recent (e.g. from a previous run)
            all_checkpoints = all_checkpoints[: all_checkpoints.index(last_checkpoint) + 1]
        elif not self._checkpoint_state_optional:
            # if last_checkpoint is not discovered from the checkpoint-state file and it isn't optional, then
            # all checkpoint files discovered must be partial or invalid, so don't return anything
            all_checkpoints.clear()
        return all_checkpoints


class CheckpointStateUpdater(object):
    """
    Class for scanning checkpoint directory and updating the checkpoint state
    """
    def __init__(self, checkpoint_dir: str, read_all: bool=False):
        """
        :param checkpoint_dir: path to checkpoint directory
        :param read_all: whether to scan the directory for existing checkpoints
        """
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_state_file = CheckpointStateFile(checkpoint_dir)
        self._all_checkpoints = list()
        # Read checkpoint state and initialize
        state_reader = CheckpointStateReader(checkpoint_dir)
        if read_all:
            self._all_checkpoints = state_reader.get_all()
        else:
            latest = state_reader.get_latest()
            if latest is not None:
                self._all_checkpoints = [latest]

    def update(self, checkpoint: SingleCheckpoint) -> None:
        """
        Update the checkpoint state with the latest checkpoint.
        :param checkpoint: SingleCheckpoint object containing name and number of checkpoint
        """
        self._all_checkpoints.append(checkpoint)
        # Simply write checkpoint_name to checkpoint-state file
        self._checkpoint_state_file.write(checkpoint)

    @property
    def last_checkpoint(self) -> Union[None, SingleCheckpoint]:
        if len(self._all_checkpoints) == 0:
            return None
        return self._all_checkpoints[-1]

    @property
    def all_checkpoints(self) -> List[SingleCheckpoint]:
        return self._all_checkpoints

    def get_checkpoint_state(self) -> Union[None, CheckpointState]:
        """
        :return: The most recent checkpoint state
        """
        if len(self._all_checkpoints) == 0:
            return None
        return CheckpointState(self._all_checkpoints, self._checkpoint_dir)


class CheckpointFilenameParser(object):
    """
    Helper object for parsing filenames that are potentially checkpoints
    """
    coach_checkpoint_filename_pattern = r'\A(([0-9]+)[^0-9])?.*?\.ckpt(-([0-9]+))?'

    def __init__(self):
        self._prog = re.compile(self.coach_checkpoint_filename_pattern)

    def parse(self, filename: str) -> Union[None, SingleCheckpoint]:
        """
        Tries to parse the filename using the checkpoint filename pattern. If successful,
        it returns tuple of (checkpoint-number, checkpoint-name). Otherwise it returns None.
        :param filename: filename to be parsed
        :return: None or (checkpoint-number, checkpoint-name)
        """
        m = self._prog.search(filename)
        if m is not None and (m.group(2) is not None or m.group(4) is not None):
            assert m.group(2) is None or m.group(4) is None  # Only one group must be valid
            checkpoint_num = int(m.group(2) if m.group(2) is not None else m.group(4))
            return SingleCheckpoint(checkpoint_num, m.group(0))
        return None


def _filter_checkpoint_files(filenames: List[str], sort_by_num: bool=True) -> List[SingleCheckpoint]:
    """
    Given a list of potential file names, return the ones that match checkpoint pattern along with
    the checkpoint number of each file name.
    :param filenames: list of all filenames
    :param sort_by_num: whether to sort the output result by checkpoint number
    :return: list of (checkpoint-number, checkpoint-filename) tuples
    """
    parser = CheckpointFilenameParser()
    checkpoints = [ckp for ckp in [parser.parse(fn) for fn in filenames] if ckp is not None]
    if sort_by_num:
        checkpoints.sort(key=lambda x: x.num)
    return checkpoints


def get_checkpoint_state(checkpoint_dir: str, all_checkpoints=False) ->Union[CheckpointState, None]:
    """
    Scan checkpoint directory and find the list of checkpoint files.
    :param checkpoint_dir: directory where checkpoints are saved
    :param all_checkpoints: if True, scan the directory and return list of all checkpoints
        as well as the most recent one
    :return: a CheckpointState for checkpoint_dir containing a sorted list of checkpoints by checkpoint-number.
        If no matching files are found, returns None.
    """
    return CheckpointStateUpdater(checkpoint_dir, read_all=all_checkpoints).get_checkpoint_state()
