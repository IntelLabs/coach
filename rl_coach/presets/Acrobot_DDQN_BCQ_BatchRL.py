import tensorflow as tf

from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, CsvDataset
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters
from rl_coach.architectures.head_parameters import QHeadParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters

from rl_coach.agents.ddqn_bcq_agent import KNNParameters

DATASET_SIZE = 50000


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

agent_params = DDQNBCQAgentParameters()
agent_params.network_wrappers['main'].batch_size = 128
# TODO cross-DL framework abstraction for a constant initializer?
agent_params.network_wrappers['main'].heads_parameters = [QHeadParameters(output_bias_initializer=tf.constant_initializer(-100))]

agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(
    100)
agent_params.algorithm.discount = 0.99

agent_params.algorithm.action_drop_method_parameters = KNNParameters()

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].softmax_temperature = 0.2

# ER size
agent_params.memory = EpisodicExperienceReplayParameters()
# DATATSET_PATH = 'acrobot.csv'
# agent_params.memory.load_memory_from_file_path = CsvDataset(DATATSET_PATH, True)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(0, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0

# Experience Generating Agent parameters
experience_generating_agent_params = DDQNAgentParameters()

# schedule parameters
experience_generating_schedule_params = ScheduleParameters()
experience_generating_schedule_params.heatup_steps = EnvironmentSteps(1000)
experience_generating_schedule_params.improve_steps = TrainingSteps(
    DATASET_SIZE - experience_generating_schedule_params.heatup_steps.num_steps)
experience_generating_schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
experience_generating_schedule_params.evaluation_steps = EnvironmentEpisodes(1)

# DQN params
experience_generating_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
experience_generating_agent_params.algorithm.discount = 0.99
experience_generating_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
experience_generating_agent_params.network_wrappers['main'].learning_rate = 0.0001
experience_generating_agent_params.network_wrappers['main'].batch_size = 128
experience_generating_agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
experience_generating_agent_params.network_wrappers['main'].heads_parameters = \
[QHeadParameters(output_bias_initializer=tf.constant_initializer(-100))]

# ER size
experience_generating_agent_params.memory = EpisodicExperienceReplayParameters()
experience_generating_agent_params.memory.max_size = \
    (MemoryGranularity.Transitions,
     experience_generating_schedule_params.heatup_steps.num_steps +
     experience_generating_schedule_params.improve_steps.num_steps + 1)

# E-Greedy schedule
experience_generating_agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, DATASET_SIZE)
experience_generating_agent_params.exploration.evaluation_epsilon = 0


################
#  Environment #
################
env_params = GymVectorEnvironment(level='Acrobot-v1')

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 150
preset_validation_params.max_episodes_to_achieve_reward = 50
preset_validation_params.read_csv_tries = 500

graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                    experience_generating_agent_params=experience_generating_agent_params,
                                    experience_generating_schedule_params=experience_generating_schedule_params,
                                    env_params=env_params,
                                    schedule_params=schedule_params,
                                    vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1),
                                    preset_validation_params=preset_validation_params,
                                    reward_model_num_epochs=30,
                                    train_to_eval_ratio=0.4)
