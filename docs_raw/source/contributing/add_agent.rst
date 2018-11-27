Adding a New Agent
==================

Coach's modularity makes adding an agent a simple and clean task.
We suggest using the following
`Jupyter notebook tutorial <https://github.com/NervanaSystems/coach/blob/master/tutorials/1.%20Implementing%20an%20Algorithm.ipynb>`_
to ramp up on this process. In general, it involves the following steps:

1. Implement your algorithm in a new file. The agent can inherit base classes such as **ValueOptimizationAgent** or
   **ActorCriticAgent**, or the more generic **Agent** base class.

   .. note::
      **ValueOptimizationAgent**, **PolicyOptimizationAgent** and **Agent** are abstract classes.
      :code:`learn_from_batch()` should be overriden with the desired behavior for the algorithm being implemented.
      If deciding to inherit from **Agent**, also :code:`choose_action()` should be overriden.

   .. code-block:: python

            def learn_from_batch(self, batch) -> Tuple[float, List, List]:
                """
                Given a batch of transitions, calculates their target values and updates the network.
                :param batch: A list of transitions
                :return: The total loss of the training, the loss per head and the unclipped gradients
                """

            def choose_action(self, curr_state):
                """
                choose an action to act with in the current episode being played. Different behavior might be exhibited when training
                 or testing.

                :param curr_state: the current state to act upon.
                :return: chosen action, some action value describing the action (q-value, probability, etc)
                """

2. Implement your agent's specific network head, if needed, at the implementation for the framework of your choice.
   For example **architectures/neon_components/heads.py**. The head will inherit the generic base class Head.
   A new output type should be added to configurations.py, and a mapping between the new head and output type should
   be defined in the get_output_head() function at **architectures/neon_components/general_network.py**

3. Define a new parameters class that inherits AgentParameters.
   The parameters class defines all the hyperparameters for the agent, and is initialized with 4 main components:

   * **algorithm**: A class inheriting AlgorithmParameters which defines any algorithm specific parameters

   * **exploration**: A class inheriting ExplorationParameters which defines the exploration policy parameters.
     There are several common exploration policies built-in which you can use, and are defined under
     the exploration sub directory. You can also define your own custom exploration policy.

   * **memory**: A class inheriting MemoryParameters which defined the memory parameters.
     There are several common memory types built-in which you can use, and are defined under the memories
     sub directory. You can also define your own custom memory.

   * **networks**: A dictionary defining all the networks that will be used by the agent. The keys of the dictionary
     define the network name and will be used to access each network through the agent class.
     The dictionary values are a class inheriting NetworkParameters, which define the network structure
     and parameters.


   Additionally, set the path property to return the path to your agent class in the following format:

   :code:`<path to python module>:<name of agent class>`

   For example,

   .. code-block:: python

            class RainbowAgentParameters(AgentParameters):
            def __init__(self):
                super().__init__(algorithm=RainbowAlgorithmParameters(),
                                 exploration=RainbowExplorationParameters(),
                                 memory=RainbowMemoryParameters(),
                                 networks={"main": RainbowNetworkParameters()})

            @property
            def path(self):
                return 'rainbow.rainbow_agent:RainbowAgent'

4. (Optional) Define a preset using the new agent type with a given environment, and the hyper-parameters that should
   be used for training on that environment.

