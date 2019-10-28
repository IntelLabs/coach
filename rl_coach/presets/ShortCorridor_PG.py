
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.agents.workshop_pg_agent import PolicyGradientsAgentParameters


N = 20
####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(2000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(N)



#########
# Agent #
#########
agent_params = PolicyGradientsAgentParameters()

agent_params.algorithm.discount = 0.99
agent_params.algorithm.apply_gradients_every_x_episodes = 5
agent_params.algorithm.num_steps_between_gradient_updates = 20000

agent_params.network_wrappers['main'].optimizer_type = 'Adam'
agent_params.network_wrappers['main'].learning_rate = 0.0005
agent_params.input_filter = NoInputFilter()
agent_params.output_filter = NoOutputFilter()


###############
# Environment #
###############

env_params = GymEnvironmentParameters(level='rl_coach.environments.toy_problems.short_corridor:ShortCorridorEnv')
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
