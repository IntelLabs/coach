from rl_coach.agents.policy_gradients_agent import PolicyGradientsAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from rl_coach.environments.gym_environment import MujocoInputFilter, Mujoco
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = PolicyGradientsAgentParameters()

agent_params.algorithm.discount = 0.99
agent_params.algorithm.apply_gradients_every_x_episodes = 5
agent_params.algorithm.num_steps_between_gradient_updates = 20000

agent_params.network_wrappers['main'].optimizer_type = 'Adam'
agent_params.network_wrappers['main'].learning_rate = 0.0005

agent_params.input_filter = MujocoInputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(1/200.))

agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
env_params = Mujoco()
env_params.level = 'CartPole-v0'

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 130
preset_validation_params.max_episodes_to_achieve_reward = 550

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)

