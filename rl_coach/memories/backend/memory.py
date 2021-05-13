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


class MemoryBackendParameters(object):

    def __init__(self, store_type, orchestrator_type, run_type, deployed: str = False):
        self.store_type = store_type
        self.orchestrator_type = orchestrator_type
        self.run_type = run_type
        self.deployed = deployed


class MemoryBackend(object):

    def __init__(self, params: MemoryBackendParameters):
        pass

    def deploy(self):
        raise NotImplemented("Not yet implemented")

    def get_endpoint(self):
        raise NotImplemented("Not yet implemented")

    def undeploy(self):
        raise NotImplemented("Not yet implemented")

    def sample(self, size: int):
        raise NotImplemented("Not yet implemented")

    def store(self, obj):
        raise NotImplemented("Not yet implemented")

    def store_episode(self, obj):
        raise NotImplemented("Not yet implemented")

    def fetch(self, num_steps=0):
        raise NotImplemented("Not yet implemented")
