from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Dense, Conv2d
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, GradientClippingMethod
from rl_coach.environments.robosuite_environment import RobosuiteEnvironmentParameters, RobosuiteLiftParameters, \
    RobosuiteRobotType, OptionalObservations
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.filters.filter import InputFilter, NoOutputFilter
from rl_coach.filters.observation import ObservationStackingFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########

# Parameters based on DDPG configuration in Surreal code:
# https://github.com/SurrealAI/surreal/blob/master/surreal/main/ddpg_configs.py

# Differences vs. Surreal:
#   1. Camera observation pre-processing should be shared between actor and critic
#   2. Exploration sigma - Surreal scales sigma linearly across actors, from 0 to max_sigma.
#      In block lifting, max_sigma = 2.0. But, if only 1 actor is used, then sigma = max_sigma / 3.0
#      Here we use the single actor setting, that is - 2/3
#   3. In the critic network in Surreal, actions are concatenated to other inputs only after first layer of middleware
#   4. Surreal uses "basic" experience replay, not episodic (I think...)

agent_params = DDPGAgentParameters()

agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter('camera', 'stacking', ObservationStackingFilter(3, concat=True))
agent_params.output_filter = NoOutputFilter()

# Exploration
agent_params.exploration = OUProcessParameters()
agent_params.exploration.sigma = 2. / 3.
agent_params.exploration.dt = 1e-3

# Hard copying every 500 updates
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(500)
agent_params.algorithm.rate_for_copying_weights_to_target = 1.0

# Camera observation pre-processing network scheme
camera_obs_scheme = [Conv2d(16, 8, 4), Conv2d(32, 4, 2), Dense(200)]

# Actor
actor_network = agent_params.network_wrappers['actor']
actor_network.input_embedders_parameters = {
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'camera': InputEmbedderParameters(scheme=camera_obs_scheme)
}
actor_network.middleware_parameters.scheme = [Dense(300), Dense(200)]
actor_network.learning_rate = 1e-4
actor_network.l2_regularization = 1e-4
actor_network.clip_gradients = 1.0
actor_network.gradients_clipping_method = GradientClippingMethod.ClipByValue
actor_network.batch_size = 512

# Critic
critic_network = agent_params.network_wrappers['critic']
critic_network.input_embedders_parameters = {
    'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'camera': InputEmbedderParameters(scheme=camera_obs_scheme)
}
critic_network.middleware_parameters.scheme = [Dense(400), Dense(300)]
critic_network.learning_rate = 1e-4
critic_network.l2_regularization = 1e-4
critic_network.batch_size = 512


###############
# Environment #
###############
env_params = RobosuiteEnvironmentParameters('lift', RobosuiteLiftParameters())
env_params.robot = RobosuiteRobotType.SAWYER
env_params.base_parameters.optional_observations = OptionalObservations.CAMERA
env_params.base_parameters.camera_depth = False
env_params.base_parameters.horizon = 200
env_params.base_parameters.ignore_done = False
env_params.frame_skip = 10

vis_params = VisualizationParameters()
vis_params.print_networks_summary = True


########
# Test #
########
preset_validation_params = PresetValidationParameters()
# preset_validation_params.trace_test_levels = ['cartpole:swingup', 'hopper:hop']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
