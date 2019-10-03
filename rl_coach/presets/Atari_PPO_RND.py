from rl_coach.agents.ppo_rnd_agent import PPORNDAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.layers import Dense
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.environments.gym_environment import atari_deterministic_v4
from rl_coach.environments.gym_exploration_environment import AtariExploration
from rl_coach.environments.environment import SingleLevelSelection

###############
# Environment #
###############
env_params = AtariExploration(level=SingleLevelSelection(atari_deterministic_v4))


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(50000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(50000000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentEpisodes(0)


#########
# Agent #
#########
agent_params = PPORNDAgentParameters()

agent_params.algorithm.rnd_sample_ratio = 1.0
agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.1
agent_params.algorithm.beta_entropy = 0.001
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 0.999
agent_params.algorithm.reward_coefficient = [2.0, 1.0]
agent_params.algorithm.discount_for_additional_value_heads = [0.99]
agent_params.algorithm.optimization_epochs = 4
agent_params.algorithm.estimate_state_value_using_gae = True
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(128 * 32)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(128 * 32)


agent_params.network_wrappers['main'].batch_size = 1024
agent_params.network_wrappers['predictor'].batch_size = 1024
agent_params.network_wrappers['constant'].batch_size = 1024
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].use_separate_networks_per_head = False
agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters(scheme=[Dense(256),
                                                                                             Dense(448),
                                                                                             Dense(448)])
agent_params.network_wrappers['predictor'].middleware_parameters = FCMiddlewareParameters(scheme=[Dense(512),
                                                                                                  Dense(512)])

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
