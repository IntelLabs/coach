from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.doom_environment import DoomEnvironmentParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)


#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()
agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/100.))
agent_params.algorithm.num_steps_between_gradient_updates = 30
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.gae_lambda = 1.0
agent_params.algorithm.beta_entropy = 0.01
agent_params.network_wrappers['main'].clip_gradients = 40.

###############
# Environment #
###############
env_params = DoomEnvironmentParameters(level='basic')

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 20
preset_validation_params.max_episodes_to_achieve_reward = 400
preset_validation_params.num_workers = 8


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
