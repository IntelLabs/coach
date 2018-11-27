from rl_coach.agents.naf_agent import NAFAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, GradientClippingMethod
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
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
agent_params = NAFAgentParameters()
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(200)]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(200)]
agent_params.network_wrappers['main'].clip_gradients = 1000
agent_params.network_wrappers['main'].gradients_clipping_method = GradientClippingMethod.ClipByValue

###############
# Environment #
###############
env_params = GymVectorEnvironment(level=SingleLevelSelection(mujoco_v2))


# this preset is currently broken - no test


########
# Test #
########
preset_validation_params = PresetValidationParameters()
# preset_validation_params.test = True
# preset_validation_params.min_reward_threshold = 200
# preset_validation_params.max_episodes_to_achieve_reward = 600
# preset_validation_params.reward_test_level = 'inverted_pendulum'
preset_validation_params.trace_test_levels = ['inverted_pendulum', 'hopper']


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
