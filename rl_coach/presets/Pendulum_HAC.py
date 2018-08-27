import numpy as np

from rl_coach.agents.hac_ddpg_agent import HACDDPGAgentParameters
from rl_coach.architectures.tensorflow_components.architecture import Dense
from rl_coach.base_parameters import VisualizationParameters, EmbeddingMergerType, EmbedderScheme
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters

from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, RunPhase, TrainingSteps
from rl_coach.environments.environment import SelectedPhaseOnlyDumpMethod
from rl_coach.environments.gym_environment import Mujoco
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.hac_graph_manager import HACGraphManager
from rl_coach.memories.episodic.episodic_hindsight_experience_replay import HindsightGoalSelectionMethod, \
    EpisodicHindsightExperienceReplayParameters
from rl_coach.memories.episodic.episodic_hrl_hindsight_experience_replay import \
    EpisodicHRLHindsightExperienceReplayParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import GoalsSpace, ReachingGoal

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(40 * 4 * 64)  # 40 epochs
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(4 * 64)  # 4 small batches of 64 episodes
schedule_params.evaluation_steps = EnvironmentEpisodes(64)
schedule_params.heatup_steps = EnvironmentSteps(0)


polar_coordinates = False

#########
# Agent #
#########

if polar_coordinates:
    distance_from_goal_threshold = np.array([0.075, 0.75])
else:
    distance_from_goal_threshold = np.array([0.075, 0.075, 0.75])
goals_space = GoalsSpace('achieved_goal',
                         ReachingGoal(default_reward=-1, goal_reaching_reward=0,
                                      distance_from_goal_threshold=distance_from_goal_threshold),
                         lambda goal, state: np.abs(goal - state))  # raw L1 distance

# top agent
top_agent_params = HACDDPGAgentParameters()

top_agent_params.memory = EpisodicHRLHindsightExperienceReplayParameters()
top_agent_params.memory.max_size = (MemoryGranularity.Transitions, 10000000)
top_agent_params.memory.hindsight_transitions_per_regular_transition = 3
top_agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
top_agent_params.memory.goals_space = goals_space
top_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(32)
top_agent_params.algorithm.num_consecutive_training_steps = 40
top_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)

# exploration - OU process
top_agent_params.exploration = OUProcessParameters()
top_agent_params.exploration.theta = 0.1

# actor
top_actor = top_agent_params.network_wrappers['actor']
top_actor.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                        'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
top_actor.middleware_parameters.scheme = [Dense([64])] * 3
top_actor.learning_rate = 0.001
top_actor.batch_size = 4096

# critic
top_critic = top_agent_params.network_wrappers['critic']
top_critic.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                         'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                         'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
top_critic.embedding_merger_type = EmbeddingMergerType.Concat
top_critic.middleware_parameters.scheme = [Dense([64])] * 3
top_critic.learning_rate = 0.001
top_critic.batch_size = 4096

# ----------

# bottom agent
bottom_agent_params = HACDDPGAgentParameters()

# TODO: we should do this is a cleaner way. probably HACGraphManager, should set this for all non top-level agents
bottom_agent_params.algorithm.in_action_space = goals_space

bottom_agent_params.memory = EpisodicHindsightExperienceReplayParameters()
bottom_agent_params.memory.max_size = (MemoryGranularity.Transitions, 12000000)
bottom_agent_params.memory.hindsight_transitions_per_regular_transition = 4
bottom_agent_params.memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future
bottom_agent_params.memory.goals_space = goals_space
bottom_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(16 * 25)  # 25 episodes is one true env episode
bottom_agent_params.algorithm.num_consecutive_training_steps = 40
bottom_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)

bottom_agent_params.exploration = EGreedyParameters()
bottom_agent_params.exploration.epsilon_schedule = ConstantSchedule(0.2)
bottom_agent_params.exploration.evaluation_epsilon = 0
bottom_agent_params.exploration.continuous_exploration_policy_parameters = OUProcessParameters()
bottom_agent_params.exploration.continuous_exploration_policy_parameters.theta = 0.1

# actor
bottom_actor = bottom_agent_params.network_wrappers['actor']
bottom_actor.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                           'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
bottom_actor.middleware_parameters.scheme = [Dense([64])] * 3
bottom_actor.learning_rate = 0.001
bottom_actor.batch_size = 4096

# critic
bottom_critic = bottom_agent_params.network_wrappers['critic']
bottom_critic.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                            'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                            'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
bottom_critic.embedding_merger_type = EmbeddingMergerType.Concat
bottom_critic.middleware_parameters.scheme = [Dense([64])] * 3
bottom_critic.learning_rate = 0.001
bottom_critic.batch_size = 4096

agents_params = [top_agent_params, bottom_agent_params]

###############
# Environment #
###############
time_limit = 1000

env_params = Mujoco()
env_params.level = "rl_coach.environments.mujoco.pendulum_with_goals:PendulumWithGoals"
env_params.additional_simulator_parameters = {"time_limit": time_limit,
                                              "random_goals_instead_of_standing_goal": False,
                                              "polar_coordinates": polar_coordinates,
                                              "goal_reaching_thresholds": distance_from_goal_threshold}
env_params.frame_skip = 10
env_params.custom_reward_threshold = -time_limit + 1

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST)]
vis_params.dump_mp4 = False
vis_params.native_rendering = False

graph_manager = HACGraphManager(agents_params=agents_params, env_params=env_params,
                                schedule_params=schedule_params, vis_params=vis_params,
                                consecutive_steps_to_run_non_top_levels=EnvironmentSteps(40))
