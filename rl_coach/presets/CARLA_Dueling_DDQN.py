import math

from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.architectures.head_parameters import DuelingQHeadParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters
from rl_coach.filters.action.box_discretization import BoxDiscretization
from rl_coach.filters.filter import OutputFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].heads_parameters = \
    [DuelingQHeadParameters(rescale_gradient_from_head_by_factor=1/math.sqrt(2))]
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty
agent_params.network_wrappers['main'].clip_gradients = 10
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)
agent_params.network_wrappers['main'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['main'].input_embedders_parameters.pop('observation')
agent_params.output_filter = OutputFilter()
agent_params.output_filter.add_action_filter('discretization', BoxDiscretization(5))

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
