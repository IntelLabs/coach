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
    def __init__(self, params: DataStoreParameters):
        pass

    def deploy(self) -> bool:
        pass

    def get_info(self):
        pass

    def undeploy(self) -> bool:
        pass

    def save_to_store(self):
        pass

    def load_from_store(self):
        pass


class SyncFiles(Enum):
    FINISHED = ".finished"
    LOCKFILE = ".lock"
