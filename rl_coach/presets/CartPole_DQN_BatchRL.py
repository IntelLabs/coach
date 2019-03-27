from copy import deepcopy

from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward import RewardRescaleFilter
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters

DATASET_SIZE = 40000

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(1)
schedule_params.evaluation_steps = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(DATASET_SIZE)

#########
# Agent #
#########
# TODO add a preset which uses a dataset to train a BatchRL graph. e.g. save a cartpole dataset in a csv format.
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].batch_size = 128

# DQN params
# agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)

# For making this become Fitted Q-Iteration we can keep the targets constant for the entire dataset size -
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(
    DATASET_SIZE / agent_params.network_wrappers['main'].batch_size)

agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(0)
agent_params.algorithm.discount = 0.98
# agent_params.algorithm.discount = 1.0


# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].l2_regularization = 0.0001
agent_params.network_wrappers['main'].softmax_temperature = 0.2
# agent_params.network_wrappers['main'].learning_rate_decay_rate = 0.95
# agent_params.network_wrappers['main'].learning_rate_decay_steps = int(DATASET_SIZE /
#                                                                   agent_params.network_wrappers['main'].batch_size)

# reward model params
agent_params.network_wrappers['reward_model'] = deepcopy(agent_params.network_wrappers['main'])
agent_params.network_wrappers['reward_model'].learning_rate = 0.0001
agent_params.network_wrappers['reward_model'].l2_regularization = 0

# ER size
agent_params.memory = EpisodicExperienceReplayParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, DATASET_SIZE)


# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(0, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0


agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/200.))

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
preset_validation_params.max_episodes_to_achieve_reward = 2000

graph_manager = BatchRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params,
                                    vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1),
                                    preset_validation_params=preset_validation_params,
                                    reward_model_num_epochs=30,
                                    train_to_eval_ratio=0.8)
