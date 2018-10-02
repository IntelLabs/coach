from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

graph_manager = BasicRLGraphManager(
    agent_params=ClippedPPOAgentParameters(),
    env_params=GymVectorEnvironment(level='CartPole-v0'),
    schedule_params=SimpleSchedule()
)

graph_manager.heatup(EnvironmentSteps(100))
graph_manager.train_and_act(EnvironmentSteps(100))