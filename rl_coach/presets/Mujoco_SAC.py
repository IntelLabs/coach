from rl_coach.agents.soft_actor_critic_agent import SoftActorCriticAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters


####################
# Graph Scheduling #
####################

# see graph_manager.py for possible schedule parameters
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(3000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(1000)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(10000)


#########
# Agent #
#########
agent_params = SoftActorCriticAgentParameters()
# override default parameters:
# value (v) networks parameters
agent_params.network_wrappers['v'].batch_size = 256
agent_params.network_wrappers['v'].learning_rate = 0.0003
agent_params.network_wrappers['v'].middleware_parameters.scheme = [Dense(256)]

# critic (q) network parameters
agent_params.network_wrappers['q'].heads_parameters[0].network_layers_sizes = (256, 256)
agent_params.network_wrappers['q'].batch_size = 256
agent_params.network_wrappers['q'].learning_rate = 0.0003

# actor (policy) network parameters
agent_params.network_wrappers['policy'].batch_size = 256
agent_params.network_wrappers['policy'].learning_rate = 0.0003
agent_params.network_wrappers['policy'].middleware_parameters.scheme = [Dense(256)]

# Input Filter
# SAC requires reward scaling for Mujoco environments.
# according to the paper:
# Hopper, Walker-2d, HalfCheetah, Ant - requires scaling of 5
# Humanoid - requires scaling of 20

agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(5))

###############
# Environment #
###############
env_params = GymVectorEnvironment(level=SingleLevelSelection(mujoco_v2))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 400
preset_validation_params.max_episodes_to_achieve_reward = 2200
preset_validation_params.reward_test_level = 'inverted_pendulum'
preset_validation_params.trace_test_levels = ['inverted_pendulum', 'hopper']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
