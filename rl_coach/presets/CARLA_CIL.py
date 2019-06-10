import os

import numpy as np
# make sure you have $CARLA_ROOT/PythonClient in your PYTHONPATH
from carla.driving_benchmark.experiment_suites import CoRL2017
from rl_coach.logger import screen

from rl_coach.agents.cil_agent import CILAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import RegressionHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.layers import Conv2d, Dense, BatchnormActivationDropout
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.filters.observation.observation_reduction_by_sub_parts_name_filter import \
    ObservationReductionBySubPartsNameFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import ImageObservationSpace
from rl_coach.utilities.carla_dataset_to_replay_buffer import create_dataset
from rl_coach.core_types import PickledReplayBuffer

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(500)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = CILAgentParameters()

# forward camera and measurements input
agent_params.network_wrappers['main'].input_embedders_parameters = {
    'CameraRGB': InputEmbedderParameters(
        scheme=[
            Conv2d(32, 5, 2),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(32, 3, 1),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(64, 3, 2),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(64, 3, 1),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(128, 3, 2),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(128, 3, 1),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(256, 3, 1),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Conv2d(256, 3, 1),
            BatchnormActivationDropout(batchnorm=True, activation_function='tanh'),
            Dense(512),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.3),
            Dense(512),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.3)
        ],
        activation_function='none'  # we define the activation function for each layer explicitly
    ),
    'measurements': InputEmbedderParameters(
         scheme=[
            Dense(128),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.5),
            Dense(128),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.5)
         ],
         activation_function='none'  # we define the activation function for each layer explicitly
    )
}

# simple fc middleware
agent_params.network_wrappers['main'].middleware_parameters = \
    FCMiddlewareParameters(
        scheme=[
            Dense(512),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.5)
        ],
        activation_function='none'
    )

# output branches
agent_params.network_wrappers['main'].heads_parameters = [
    RegressionHeadParameters(
        scheme=[
            Dense(256),
            BatchnormActivationDropout(activation_function='tanh', dropout_rate=0.5),
            Dense(256),
            BatchnormActivationDropout(activation_function='tanh')
        ],
        num_output_head_copies=4  # follow lane, left, right, straight
    )
]
# TODO: there should be another head predicting the speed which is connected directly to the forward camera embedding

agent_params.network_wrappers['main'].batch_size = 120
agent_params.network_wrappers['main'].learning_rate = 0.0002


# crop and rescale the image + use only the forward speed measurement
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter('CameraRGB', 'cropping',
                                                 ObservationCropFilter(crop_low=np.array([115, 0, 0]),
                                                                       crop_high=np.array([510, -1, -1])))
agent_params.input_filter.add_observation_filter('CameraRGB', 'rescale',
                                                 ObservationRescaleToSizeFilter(
                                                     ImageObservationSpace(np.array([88, 200, 3]), high=255)))
agent_params.input_filter.add_observation_filter('CameraRGB', 'to_uint8', ObservationToUInt8Filter(0, 255))
agent_params.input_filter.add_observation_filter(
    'measurements', 'select_speed',
    ObservationReductionBySubPartsNameFilter(
        ["forward_speed"], reduction_method=ObservationReductionBySubPartsNameFilter.ReductionMethod.Keep))

# no exploration is used
agent_params.exploration = AdditiveNoiseParameters()
agent_params.exploration.noise_schedule = ConstantSchedule(0)
agent_params.exploration.evaluation_noise = 0

# no playing during the training phase
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(0)

# use the following command line to download and extract the CARLA dataset:
# python rl_coach/utilities/carla_dataset_to_replay_buffer.py
agent_params.memory.load_memory_from_file_path = PickledReplayBuffer("./datasets/carla_train_set_replay_buffer.p")
agent_params.memory.state_key_with_the_class_index = 'high_level_command'
agent_params.memory.num_classes = 4

# download dataset if it doesn't exist
if not os.path.exists(agent_params.memory.load_memory_from_file_path):
    screen.log_title("The CARLA dataset is not present in the following path: {}"
                     .format(agent_params.memory.load_memory_from_file_path))
    result = screen.ask_yes_no("Do you want to download it now?")
    if result:
        create_dataset(None, "./datasets/carla_train_set_replay_buffer.p")
    else:
        screen.error("Please update the path to the CARLA dataset in the CARLA_CIL preset", crash=True)


###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.cameras = ['CameraRGB']
env_params.camera_height = 600
env_params.camera_width = 800
env_params.separate_actions_for_throttle_and_brake = True
env_params.allow_braking = True
env_params.quality = CarlaEnvironmentParameters.Quality.EPIC
env_params.experiment_suite = CoRL2017('Town01')

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
