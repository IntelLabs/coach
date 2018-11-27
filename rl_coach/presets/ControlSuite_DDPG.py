from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.control_suite_environment import ControlSuiteEnvironmentParameters, control_suite_envs
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
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
agent_params = DDPGAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['measurements'] = \
    agent_params.network_wrappers['actor'].input_embedders_parameters.pop('observation')
agent_params.network_wrappers['critic'].input_embedders_parameters['measurements'] = \
    agent_params.network_wrappers['critic'].input_embedders_parameters.pop('observation')
agent_params.network_wrappers['actor'].input_embedders_parameters['measurements'].scheme = [Dense(300)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(200)]
agent_params.network_wrappers['critic'].input_embedders_parameters['measurements'].scheme = [Dense(400)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter("rescale", RewardRescaleFilter(1/10.))

###############
# Environment #
###############
env_params = ControlSuiteEnvironmentParameters(level=SingleLevelSelection(control_suite_envs))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['cartpole:swingup', 'hopper:hop']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
