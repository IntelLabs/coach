
import numpy as np
from typing import List
from rl_coach.core_types import ActionType
from rl_coach.spaces import ActionSpace
from rl_coach.exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters


class MyExplorationPolicy(ExplorationPolicy):
    """
    An exploration policy takes the predicted actions or action values from the agent, and selects the action to
    actually apply to the environment using some predefined algorithm.
    """
    def __init__(self, action_space: ActionSpace):
        #self.phase = RunPhase.HEATUP
        self.action_space = action_space
        super().__init__(action_space)

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        if (np.random.rand() < 0.5):
            chosen_action = self.action_space.sample()
        else:
            chosen_action = np.argmax(action_values)
        probabilities = np.zeros(len(self.action_space.actions))
        probabilities[chosen_action] = 1
        return chosen_action, probabilities

    def get_control_param(self):
        return 0



class MyExplorationParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()

    @property
    def path(self):
        return 'exploration:MyExplorationPolicy'
