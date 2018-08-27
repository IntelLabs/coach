from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import RunPhase
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from rl_coach.environments.starcraft2_environment import StarCraft2EnvironmentParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import ConstantSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.gae_lambda = 0.96
agent_params.algorithm.beta_entropy = 0

agent_params.network_wrappers['main'].clip_gradients = 10.0
agent_params.network_wrappers['main'].learning_rate = 0.00001
# agent_params.network_wrappers['main'].batch_size = 20
agent_params.network_wrappers['main'].input_embedders_parameters = {
    "screen": InputEmbedderParameters(input_rescaling={'image': 3.0})
}

agent_params.exploration = AdditiveNoiseParameters()
agent_params.exploration.noise_percentage_schedule = ConstantSchedule(0.05)
# agent_params.exploration.noise_percentage_schedule = LinearSchedule(0.4, 0.05, 100000)
agent_params.exploration.evaluation_noise_percentage = 0.05

agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

###############
# Environment #
###############

env_params = StarCraft2EnvironmentParameters()
env_params.level = 'CollectMineralShards'
env_params.feature_screen_maps_to_use = [5]
env_params.feature_minimap_maps_to_use = [5]

vis_params = VisualizationParameters()
# vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST),MaxDumpMethod()]
vis_params.dump_mp4 = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
