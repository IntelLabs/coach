from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, fetch_v1
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_clipping_filter import ObservationClippingFilter
from rl_coach.filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.episodic.episodic_hindsight_experience_replay import EpisodicHindsightExperienceReplayParameters, \
    HindsightGoalSelectionMethod
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import GoalsSpace, ReachingGoal

cycles = 100  # 20 for reach. for others it's 100

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(cycles * 200)  # 200 epochs
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(cycles)  # 50 cycles
schedule_params.evaluation_steps = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = DDPGAgentParameters()

# actor
actor_network = agent_params.network_wrappers['actor']
actor_network.learning_rate = 0.001
actor_network.batch_size = 256
actor_network.optimizer_epsilon = 1e-08
actor_network.adam_optimizer_beta1 = 0.9
actor_network.adam_optimizer_beta2 = 0.999
actor_network.input_embedders_parameters = {
    'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)
}
actor_network.middleware_parameters = FCMiddlewareParameters(scheme=[Dense(256), Dense(256), Dense(256)])
actor_network.heads_parameters[0].batchnorm = False

# critic
critic_network = agent_params.network_wrappers['critic']
critic_network.learning_rate = 0.001
critic_network.batch_size = 256
critic_network.optimizer_epsilon = 1e-08
critic_network.adam_optimizer_beta1 = 0.9
critic_network.adam_optimizer_beta2 = 0.999
critic_network.input_embedders_parameters = {
    'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty)
}
critic_network.middleware_parameters = FCMiddlewareParameters(scheme=[Dense(256), Dense(256), Dense(256)])

agent_params.algorithm.discount = 0.98
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(1)
agent_params.algorithm.num_consecutive_training_steps = 40
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)
agent_params.algorithm.rate_for_copying_weights_to_target = 0.05
agent_params.algorithm.action_penalty = 1
agent_params.algorithm.use_non_zero_discount_for_terminal_states = True
agent_params.algorithm.clip_critic_targets = [-50, 0]

# HER parameters
agent_params.memory = EpisodicHindsightExperienceReplayParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10**6)
agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
agent_params.memory.hindsight_transitions_per_regular_transition = 4
agent_params.memory.goals_space = GoalsSpace(goal_name='achieved_goal',
                                             reward_type=ReachingGoal(distance_from_goal_threshold=0.05,
                                                                      goal_reaching_reward=0,
                                                                      default_reward=-1),
                                             distance_metric=GoalsSpace.DistanceMetric.Euclidean)
agent_params.memory.shared_memory = True

# exploration parameters
agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.3)
agent_params.exploration.evaluation_epsilon = 0
# they actually take the noise_percentage_schedule to be 0.2 * max_abs_range which is 0.1 * total_range
agent_params.exploration.continuous_exploration_policy_parameters.noise_percentage_schedule = ConstantSchedule(0.1)
agent_params.exploration.continuous_exploration_policy_parameters.evaluation_noise_percentage = 0

agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter('observation', 'clipping', ObservationClippingFilter(-200, 200))

agent_params.pre_network_filter = InputFilter()
agent_params.pre_network_filter.add_observation_filter('observation', 'normalize_observation',
                                                       ObservationNormalizationFilter(name='normalize_observation'))
agent_params.pre_network_filter.add_observation_filter('achieved_goal', 'normalize_achieved_goal',
                                                       ObservationNormalizationFilter(name='normalize_achieved_goal'))
agent_params.pre_network_filter.add_observation_filter('desired_goal', 'normalize_desired_goal',
                                                       ObservationNormalizationFilter(name='normalize_desired_goal'))

###############
# Environment #
###############
env_params = GymVectorEnvironment(level=SingleLevelSelection(fetch_v1))
env_params.custom_reward_threshold = -49

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['slide', 'pick_and_place', 'push', 'reach']


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)

