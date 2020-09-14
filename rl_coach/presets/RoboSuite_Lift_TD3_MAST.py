from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.agents.td3_agent import TD3AgentParameters
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Dense, Conv2d, BatchnormActivationDropout, Flatten
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, GradientClippingMethod, \
    EveryNEpisodesDumpFilter, MaxDumpFilter, SelectedPhaseOnlyDumpFilter, RunPhase, TaskIdDumpFilter
from rl_coach.environments.robosuite_environment import RobosuiteEnvironmentParameters, OptionalObservations, \
    robosuite_environments
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.filters.filter import InputFilter, NoOutputFilter, NoInputFilter
from rl_coach.filters.observation import ObservationStackingFilter, ObservationRGBToYFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
from rl_coach.graph_managers.mast_graph_manager import MASTGraphManager

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(10000)

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

agent_params = TD3AgentParameters()
agent_params.algorithm.use_non_zero_discount_for_terminal_states = True

agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(200)
agent_params.algorithm.mast_trainer_publish_policy_every_num_fetched_steps = EnvironmentSteps(200)

agent_params.input_filter = InputFilter()

agent_params.input_filter.add_observation_filter('camera', 'grayscale', ObservationRGBToYFilter())
agent_params.input_filter.add_observation_filter('camera', 'stacking', ObservationStackingFilter(3, concat=False))

agent_params.output_filter = NoOutputFilter()

# Camera observation pre-processing network scheme
camera_obs_scheme = [
    Conv2d(16, 8, 4),
    BatchnormActivationDropout(activation_function='relu'),
    Conv2d(32, 4, 2),
    BatchnormActivationDropout(activation_function='relu'),
    Flatten(),
    Dense(256),
    BatchnormActivationDropout(activation_function='relu')
]

# Actor
actor_network = agent_params.network_wrappers['actor']
actor_network.input_embedders_parameters = {
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'camera': InputEmbedderParameters(scheme=camera_obs_scheme, activation_function='none')
}

actor_network.middleware_parameters.scheme = [Dense(300), Dense(200)]
actor_network.learning_rate = 3e-4
# actor_network.l2_regularization = 1e-4
# actor_network.clip_gradients = 10.0
# actor_network.gradients_clipping_method = GradientClippingMethod.ClipByValue
# actor_network.batch_size = 512

# Critic
critic_network = agent_params.network_wrappers['critic']
critic_network.input_embedders_parameters = {
    'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'camera': InputEmbedderParameters(scheme=camera_obs_scheme, activation_function='none')
}
critic_network.middleware_parameters.scheme = [Dense(400), Dense(300)]
critic_network.learning_rate = 3e-4
# critic_network.l2_regularization = 1e-4
# critic_network.batch_size = 512


###############
# Environment #
###############
env_params = RobosuiteEnvironmentParameters(level=SingleLevelSelection(robosuite_environments, force_lower=False))
env_params.robot = 'PandaLab'
# env_params.controller = 'JOINT_VELOCITY'
env_params.controller = 'IK_POSE_POS'
env_params.base_parameters.optional_observations = OptionalObservations.CAMERA
env_params.base_parameters.render_camera = 'frontview'
env_params.base_parameters.camera_names = 'labview'
env_params.base_parameters.camera_depths = False
env_params.base_parameters.horizon = 200
env_params.base_parameters.ignore_done = False
env_params.frame_skip = 1
env_params.base_parameters.control_freq = 2

# Use extra_parameters for any Robosuite parameter not exposed by RobosuiteBaseParameters
# These are mostly task-specific parameters. For example, for the "lift" task once could modify
# the table size:
# env_params.extra_parameters = {'table_full_size': (0.5, 0.5, 0.5)}
env_params.extra_parameters = {}

vis_params = VisualizationParameters()
vis_params.dump_mp4 = True
vis_params.video_dump_filters = [[EveryNEpisodesDumpFilter(5), MaxDumpFilter()],
                                 [TaskIdDumpFilter(0), SelectedPhaseOnlyDumpFilter(RunPhase.TEST)]]
vis_params.print_networks_summary = True


########
# Test #
########
preset_validation_params = PresetValidationParameters()
# preset_validation_params.trace_test_levels = ['cartpole:swingup', 'hopper:hop']

graph_manager = MASTGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
