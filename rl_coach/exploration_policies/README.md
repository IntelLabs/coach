# Exploration Policy

An exploration policy is a module that is responsible for choosing the action according to the action values, the
current phase, its internal state and the specific exploration policy algorithm.

A custom exploration policy should implement both the exploration policy class and the exploration policy parameters
class, which defines the parameters and the location of the exploration policy module.
The parameters of the exploration policy class should match the parameters in the exploration policy parameters class.

Exploration policies typically have some control parameter that defines its current exploration state, and
a schedule for this parameter. This schedule can be defined using the Schedule class which is defined in
exploration_policy.py.

A custom implementation should look as follows:

```
class CustomExplorationParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        ...

    @property
    def path(self):
        return 'module_path:class_name'


class CustomExplorationPolicy(ExplorationPolicy):
    def __init__(self, action_space: ActionSpace, ...):
        super().__init__(action_space)

    def reset(self):
        ...

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        ...

    def change_phase(self, phase):
        ...

    def get_control_param(self):
        ...
```