from collections import OrderedDict

from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.filters.action import BoxDiscretization
from rl_coach.filters.filter import OutputFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.agents.wolpertinger_agent import WolpertingerAgentParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(2000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(3000)

#########
# Agent #
#########
agent_params = WolpertingerAgentParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [Dense(400)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [Dense(400)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(300)]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty
agent_params.output_filter = \
    OutputFilter(
        action_filters=OrderedDict([
            ('discretization', BoxDiscretization(num_bins_per_dimension=int(1e6)))
        ]),
        is_a_reference_filter=False
    )

###############
# Environment #
###############
env_params = GymVectorEnvironment(level=SingleLevelSelection(mujoco_v2))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 500
preset_validation_params.max_episodes_to_achieve_reward = 1000
preset_validation_params.reward_test_level = 'inverted_pendulum'
preset_validation_params.trace_test_levels = ['inverted_pendulum']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
