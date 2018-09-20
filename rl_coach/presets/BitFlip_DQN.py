from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, \
    PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import ConstantSchedule

bit_length = 8

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(400000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(16 * 50)  # 50 cycles
schedule_params.evaluation_steps = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = DQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.001
agent_params.network_wrappers['main'].batch_size = 128
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(256)]
agent_params.network_wrappers['main'].input_embedders_parameters = {
    'state': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)
}
agent_params.algorithm.discount = 0.98
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(16)
agent_params.algorithm.num_consecutive_training_steps = 40
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)
agent_params.algorithm.rate_for_copying_weights_to_target = 0.05
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10**6)
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.2)
agent_params.exploration.evaluation_epsilon = 0

###############
# Environment #
###############
env_params = GymVectorEnvironment(level='rl_coach.environments.toy_problems.bit_flip:BitFlip')
env_params.additional_simulator_parameters = {'bit_length': bit_length, 'mean_zero': True}
env_params.custom_reward_threshold = -bit_length + 1


########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = -7.9
preset_validation_params.max_episodes_to_achieve_reward = 10000

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)


# self.algorithm.add_intrinsic_reward_for_reaching_the_goal = False

