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
import time
import os

from rl_coach.checkpoint import CheckpointStateReader
from rl_coach.data_stores.data_store import SyncFiles


class CheckpointDataStore(object):
    """
    A DataStore which relies on the GraphManager check pointing methods to communicate policies.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_num = 0

    def end_of_policies(self) -> bool:
        """
        Returns true if no new policies will be added to this DataStore. This typically happens
        because training has completed and is used to signal to the rollout workers to stop.
        """
        return os.path.exists(
            os.path.join(self.checkpoint_dir, SyncFiles.FINISHED.value)
        )

    def save_policy(self, graph_manager):
        # TODO: it would be nice if restore_checkpoint accepted a checkpoint path as a
        # parameter. as it is, one cannot distinguish between checkpoints used for coordination
        # and checkpoints requested to a persistent disk for later use
        graph_manager.task_parameters.checkpoint_restore_path = self.checkpoint_dir
        graph_manager.save_checkpoint()

    def load_policy(self, graph_manager, require_new_policy=True, timeout=None):
        """
        Load a checkpoint into the specified graph_manager. The expectation here is that
        save_to_store() and load_from_store() will synchronize a checkpoint directory with a
        central repository such as NFS or S3.

        :param graph_manager: the graph_manager to load the policy into
        :param require_new_policy: if True, only load a policy if it hasn't been loaded in this
        process yet before.
        :param timeout: Will only try to load the policy once if timeout is None, otherwise will
        retry for timeout seconds
        """
        if self._new_policy_exists(require_new_policy, timeout):
            # TODO: it would be nice if restore_checkpoint accepted a checkpoint path as a
            # parameter. as it is, one cannot distinguish between checkpoints used for coordination
            # and checkpoints requested to a persistent disk for later use
            graph_manager.task_parameters.checkpoint_restore_path = self.checkpoint_dir
            graph_manager.restore_checkpoint()

    def _new_policy_exists(self, require_new_policy=True, timeout=None) -> bool:
        """
        :param require_new_policy: if True, only load a policy if it hasn't been loaded in this
        process yet before.
        :param timeout: Will only try to load the policy once if timeout is None, otherwise will
        retry for timeout seconds
        """
        checkpoint_state_reader = CheckpointStateReader(
            self.checkpoint_dir, checkpoint_state_optional=False
        )
        checkpoint = "first"
        if timeout is None:
            timeout = 0
        timeout_ends = time.time() + timeout
        while time.time() < timeout_ends or checkpoint == "first":
            if self.end_of_policies():
                return False

            self.load_from_store()

            checkpoint = checkpoint_state_reader.get_latest()
            if checkpoint is not None:
                if not require_new_policy or checkpoint.num > self.checkpoint_num:
                    self.checkpoint_num = checkpoint.num
                    return True

        raise ValueError(
            "Waited for {timeout} seconds, but no first policy was received.".format(
                timeout=timeout
            )
        )
