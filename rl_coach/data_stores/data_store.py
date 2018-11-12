
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
