from copy import deepcopy
from rl_coach.agents.dqn_agent import DQNAgentParameters

from rl_coach.architectures.tensorflow_components.layers import Dense
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
from rl_coach.architectures.head_parameters import ClassificationHeadParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters

from rl_coach.agents.ddqn_bcq_agent import KNNParameters
from rl_coach.agents.ddqn_bcq_agent import NNImitationModelParameters

DATASET_SIZE = 10000

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

# using a set of 'unstable' hyper-params to showcase the value of BCQ. Using the same hyper-params with standard DDQN
# will cause Q values to unboundedly increase, and the policy convergence to be unstable.
agent_params = DDQNBCQAgentParameters()
agent_params.network_wrappers['main'].batch_size = 128
# agent_params.network_wrappers['main'].batch_size = 1024

# DQN params

# For making this become Fitted Q-Iteration we can keep the targets constant for the entire dataset size -
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(
    DATASET_SIZE / agent_params.network_wrappers['main'].batch_size)
#
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(
    100)
# agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)

agent_params.algorithm.discount = 0.98

# can use either a kNN or a NN based model for predicting which actions not to max over in the bellman equation
agent_params.algorithm.action_drop_method_parameters = KNNParameters()
# agent_params.algorithm.action_drop_method_parameters = NNImitationModelParameters()
# agent_params.algorithm.action_drop_method_parameters.imitation_model_num_epochs = 500

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].l2_regularization = 0.0001
agent_params.network_wrappers['main'].softmax_temperature = 0.2

# reward model params
agent_params.network_wrappers['reward_model'] = deepcopy(agent_params.network_wrappers['main'])
agent_params.network_wrappers['reward_model'].learning_rate = 0.0001
agent_params.network_wrappers['reward_model'].l2_regularization = 0

agent_params.network_wrappers['imitation_model'] = deepcopy(agent_params.network_wrappers['main'])
agent_params.network_wrappers['imitation_model'].learning_rate = 0.0001
agent_params.network_wrappers['imitation_model'].l2_regularization = 0

agent_params.network_wrappers['imitation_model'].heads_parameters = [ClassificationHeadParameters()]
agent_params.network_wrappers['imitation_model'].input_embedders_parameters['observation'].scheme = \
    [Dense(1024), Dense(1024), Dense(512), Dense(512), Dense(256)]
agent_params.network_wrappers['imitation_model'].middleware_parameters.scheme = [Dense(128), Dense(64)]


# ER size
agent_params.memory = EpisodicExperienceReplayParameters()

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(0, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0

# Input filtering
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/200.))




# Experience Generating Agent parameters
experience_generating_agent_params = DQNAgentParameters()

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
experience_generating_agent_params.network_wrappers['main'].learning_rate = 0.00025
experience_generating_agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

# ER size
experience_generating_agent_params.memory = EpisodicExperienceReplayParameters()
experience_generating_agent_params.memory.max_size = \
    (MemoryGranularity.Transitions,
     experience_generating_schedule_params.heatup_steps.num_steps +
     experience_generating_schedule_params.improve_steps.num_steps)

# E-Greedy schedule
experience_generating_agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)


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
