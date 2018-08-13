# Block Factory

The block factory is a class which creates a block that fits into a specific RL scheme.
Example RL schemes are: self play, multi agent, HRL, basic RL, etc.
The block factory should create all the components of the block and return the block scheduler.
The block factory will then be used to create different combinations of components.
For example, an HRL factory can be later instantiated with:
* env = Atari Breakout
* master (top hierarchy level) agent = DDPG
* slave (bottom hierarchy level) agent = DQN

A custom block factory implementation should look as follows:

```
class CustomFactory(BlockFactory):
    def __init__(self, custom_params):
        super().__init__()

    def _create_block(self, task_index: int, device=None) -> BlockScheduler:
        """
        Create all the block modules and the block scheduler
        :param task_index: the index of the process on which the worker will be run
        :return: the initialized block scheduler
        """

        # Create env
        # Create composite agents
        # Create level managers
        # Create block scheduler

        return block_scheduler
```