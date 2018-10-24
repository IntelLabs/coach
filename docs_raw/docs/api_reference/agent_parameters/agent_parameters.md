# Agent Parameters

Each agent should have a dedicated parameters class which inherits from AgentParameters, and instantiates it with
the following parameters:

algorithm: (AlgorithmParameters)
"""
:param algorithm: the algorithmic parameters
:param exploration: the exploration policy parameters
:param memory: the memory module parameters
:param networks: the parameters for the networks of the agent
:param visualization: the visualization parameters
"""