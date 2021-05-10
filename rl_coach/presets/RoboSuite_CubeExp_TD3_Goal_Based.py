from rl_coach.agents.td3_exp_agent import TD3GoalBasedAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Dense, Conv2d, BatchnormActivationDropout, Flatten
from rl_coach.base_parameters import VisualizationParameters, EmbedderScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.robosuite_environment import RobosuiteGoalBasedExpEnvironmentParameters, \
    OptionalObservations, robosuite_environments
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.architectures.head_parameters import RNDHeadParameters

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(300000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(300000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########

agent_params = TD3GoalBasedAgentParameters()
agent_params.algorithm.use_non_zero_discount_for_terminal_states = False

agent_params.input_filter = NoInputFilter()

agent_params.output_filter = NoOutputFilter()

# Camera observation pre-processing network scheme
camera_obs_scheme = [
    Conv2d(32, 8, 4),
    BatchnormActivationDropout(activation_function='relu'),
    Conv2d(64, 4, 2),
    BatchnormActivationDropout(activation_function='relu'),
    Conv2d(64, 3, 1),
    BatchnormActivationDropout(activation_function='relu'),
    Flatten(),
    Dense(256),
    BatchnormActivationDropout(activation_function='relu')
]

obs_name = 'obs-goal'

# Actor
actor_network = agent_params.network_wrappers['actor']
actor_network.input_embedders_parameters = {
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    obs_name: InputEmbedderParameters(scheme=camera_obs_scheme, activation_function='none')
}

actor_network.middleware_parameters.scheme = [Dense(300), Dense(200)]
actor_network.learning_rate = 1e-4

# Critic
critic_network = agent_params.network_wrappers['critic']
critic_network.input_embedders_parameters = {
    'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    'measurements': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    obs_name: InputEmbedderParameters(scheme=camera_obs_scheme, activation_function='none')
}

critic_network.middleware_parameters.scheme = [Dense(400), Dense(300)]
critic_network.learning_rate = 1e-4

# RND
agent_params.network_wrappers['predictor'].input_embedders_parameters = \
    {'camera': InputEmbedderParameters(scheme=EmbedderScheme.Empty,
                                       input_rescaling={'image': 1.0},
                                       flatten=False)}
agent_params.network_wrappers['constant'].input_embedders_parameters = \
    {'camera': InputEmbedderParameters(scheme=EmbedderScheme.Empty,
                                       input_rescaling={'image': 1.0},
                                       flatten=False)}
agent_params.network_wrappers['predictor'].heads_parameters = [RNDHeadParameters(is_predictor=True)]

###############
# Environment #
###############
env_params = RobosuiteGoalBasedExpEnvironmentParameters(
    level=SingleLevelSelection(robosuite_environments, force_lower=False)
)
env_params.robot = 'PandaLab'
env_params.controller = 'IK_POSE_POS'
env_params.base_parameters.optional_observations = OptionalObservations.CAMERA
env_params.base_parameters.render_camera = 'frontview'
env_params.base_parameters.camera_names = 'labview'
env_params.base_parameters.camera_depths = False
env_params.base_parameters.horizon = 200
env_params.base_parameters.ignore_done = False
env_params.base_parameters.use_object_obs = True
env_params.frame_skip = 1
env_params.base_parameters.control_freq = 2

size = 100
env_params.base_parameters.camera_heights = size
env_params.base_parameters.camera_widths = size
env_params.extra_parameters = {'camera_crop_boxes': (size - 84, size // 2 - 38, 84, 84)}

vis_params = VisualizationParameters()


########
# Test #
########
preset_validation_params = PresetValidationParameters()


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
