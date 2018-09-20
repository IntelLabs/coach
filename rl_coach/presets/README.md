# Defining Presets

In Coach, we use a Preset mechanism in order to define reproducible experiments.
A Preset defines all the parameters of an experiment in a single file, and can be executed from the command
line using the file name.
Presets can be very simple by using the default parameters of the algorithm and environment.
They can also be explicit and define all the parameters in order to avoid hidden logic.
The outcome of a preset is a GraphManager.


Let's start with the simplest preset possible.
We will define a preset for training the CartPole environment using Clipped PPO.
The 3 minimal things we need to define in each preset are the agent, the environment and a schedule.

```
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

graph_manager = BasicRLGraphManager(
    agent_params=ClippedPPOAgentParameters(),
    env_params=GymVectorEnvironment(level='CartPole-v0'),
    schedule_params=SimpleSchedule()
)
```

Most presets in Coach are much more explicit than this. The motivation behind this is to be as transparent as
possible regarding all the changes needed relative to the basic parameters defined in the algorithm paper.