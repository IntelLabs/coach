


class DeployParameters(object):

    def __init__(self):
        pass


class Deploy(object):

    def __init__(self, deploy_parameters):
        self.deploy_parameters = deploy_parameters

    def setup(self) -> bool:
        pass

    def deploy(self) -> bool:
        pass