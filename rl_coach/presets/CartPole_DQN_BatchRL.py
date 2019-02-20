from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(1)
schedule_params.evaluation_steps = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(40000)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(0)
agent_params.algorithm.discount = 0.99


# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].l2_regularization = 0.1
agent_params.network_wrappers['main'].batch_size = 256
# agent_params.network_wrappers['main'].learning_rate_decay_rate = 0.5
# agent_params.network_wrappers['main'].learning_rate_decay_steps = int(40000 /
#                                                                   agent_params.network_wrappers['main'].batch_size)

# ER size
agent_params.memory = EpisodicExperienceReplayParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

# E-Greedy schedule
# agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)
agent_params.exploration.epsilon_schedule = LinearSchedule(0, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0

################
#  Environment #
################
env_params = GymVectorEnvironment(level='CartPole-v0')

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 150
preset_validation_params.max_episodes_to_achieve_reward = 250

graph_manager = BatchRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)