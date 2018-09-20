from rl_coach.agents.dfp_agent import DFPAgentParameters
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, MiddlewareScheme, \
    PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, EnvironmentEpisodes
from rl_coach.environments.doom_environment import DoomEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(6250000)
# original paper evaluates according to these. But, this preset converges significantly faster - can be evaluated
# much often.
# schedule_params.steps_between_evaluation_periods = EnvironmentSteps(62500)
# schedule_params.evaluation_steps = EnvironmentSteps(6250)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(5)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)

# There is no heatup for DFP. heatup length is determined according to batch size. See below.

#########
# Agent #
#########
agent_params = DFPAgentParameters()
schedule_params.heatup_steps = EnvironmentSteps(agent_params.network_wrappers['main'].batch_size)

agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.algorithm.goal_vector = [1]  # health

# this works better than the default which is set to 8 (while running with 8 workers)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# scale observation and measurements to be -0.5 <-> 0.5
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].input_rescaling['vector'] = 100.
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].input_offset['vector'] = 0.5
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].input_offset['vector'] = 0.5

# changing the network scheme to match Coach's default network, as it performs better on this preset
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['goal'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Medium

# scale the target measurements according to the paper (dividing by standard deviation)
agent_params.algorithm.scale_measurements_targets['GameVariable.HEALTH'] = 30.0

###############
# Environment #
###############
env_params = DoomEnvironmentParameters(level='HEALTH_GATHERING_SUPREME_COACH_LOCAL')


########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test_using_a_trace_test = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
