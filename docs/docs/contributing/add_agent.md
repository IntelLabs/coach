<!-- language-all: python -->

Coach's modularity makes adding an agent a simple and clean task, that involves the following steps:

1.	Implement your algorithm in a new file under the agents directory. The agent can inherit base classes such as **ValueOptimizationAgent** or **ActorCriticAgent**, or the more generic **Agent** base class.
    
    * **ValueOptimizationAgent**, **PolicyOptimizationAgent** and **Agent** are abstract classes. 
    learn_from_batch() should be overriden with the desired behavior for the algorithm being implemented. If deciding to inherit from **Agent**, also choose_action() should be overriden.       
        
    
            def learn_from_batch(self, batch):
                """
                Given a batch of transitions, calculates their target values and updates the network.
                :param batch: A list of transitions
                :return: The loss of the training
                """
                pass
                
            def choose_action(self, curr_state, phase=RunPhase.TRAIN):
                """
                choose an action to act with in the current episode being played. Different behavior might be exhibited when training
                 or testing.
                 
                :param curr_state: the current state to act upon.  
                :param phase: the current phase: training or testing.
                :return: chosen action, some action value describing the action (q-value, probability, etc)
                """
                pass
                
            
       
    * Make sure to add your new agent to **agents/\_\_init\_\_.py**
    
2.	Implement your agent's specific network head, if needed, at the implementation for the framework of your choice. For example **architectures/neon_components/heads.py**. The head will inherit the generic base class Head.
    A new output type should be added to configurations.py, and a mapping between the new head and output type should be defined in the get_output_head() function at **architectures/neon_components/general_network.py**
3.	Define a new configuration class at configurations.py, which includes the new agent name in the **type** field, the new output type in the **output_types** field, and assigning default values to hyperparameters.
4.	(Optional) Define a preset using the new agent type with a given environment, and the hyperparameters that should be used for training on that environment.

