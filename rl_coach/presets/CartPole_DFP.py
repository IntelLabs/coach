from rl_coach.agents.dfp_agent import DFPAgentParameters, HandlingTargetsAfterEpisodeEnd
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(100)


#########
# Agent #
#########
agent_params = DFPAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['goal'].scheme = EmbedderScheme.Medium
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].scheme = EmbedderScheme.Medium
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0.01, 3000)
agent_params.exploration.evaluation_epsilon = 0.01
agent_params.algorithm.discount = 1.0
agent_params.algorithm.use_accumulated_reward_as_measurement = True
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.algorithm.goal_vector = [1]  # accumulated_reward
agent_params.algorithm.handling_targets_after_episode_end = HandlingTargetsAfterEpisodeEnd.LastStep

###############
# Environment #
###############
env_params = GymVectorEnvironment(level='CartPole-v0')

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 120
preset_validation_params.max_episodes_to_achieve_reward = 250

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
