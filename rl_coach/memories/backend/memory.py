

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
