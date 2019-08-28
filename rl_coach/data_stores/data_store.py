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


from enum import Enum


class DataStoreParameters(object):
    def __init__(self, store_type, orchestrator_type, orchestrator_params):
        self.store_type = store_type
        self.orchestrator_type = orchestrator_type
        self.orchestrator_params = orchestrator_params


class DataStore(object):
    """
    DataStores are used primarily to synchronize policies between training workers and rollout
    workers. In the case of the S3DataStore, it is also being used to explicitly log artifacts such
    as videos and logs into s3 for users to look at later. Artifact logging should be moved into a
    separate instance of the DataStore class, or a different class altogether. It is possible that
    users might be interested in logging artifacts through s3, but coordinating communication of
    policies using something else like redis.
    """

    def __init__(self, params: DataStoreParameters):
        """
        The parameters provided in the constructor to a DataStore are expected to contain the
        parameters necessary to serialize and deserialize this DataStore.
        """
        raise NotImplementedError()

    def deploy(self) -> bool:
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

    def undeploy(self) -> bool:
        raise NotImplementedError()

    def save_to_store(self):
        raise NotImplementedError()

    def load_from_store(self):
        raise NotImplementedError()

    def save_policy(self, graph_manager):
        raise NotImplementedError()

    def load_policy(self, graph_manager, timeout=-1):
        raise NotImplementedError()

    def end_of_policies(self) -> bool:
        raise NotImplementedError()

    def setup_checkpoint_dir(self, crd=None):
        pass


class SyncFiles(Enum):
    FINISHED = ".finished"
    LOCKFILE = ".lock"
    TRAINER_READY = ".ready"
