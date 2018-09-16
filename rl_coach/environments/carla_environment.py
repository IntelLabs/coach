#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import random
import sys
from os import path, environ

from rl_coach.logger import screen
from rl_coach.filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter

try:
    if 'CARLA_ROOT' in environ:
        sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
    else:
        screen.error("CARLA_ROOT was not defined. Please set it to point to the CARLA root directory and try again.")
    from carla.client import CarlaClient
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.sensor import Camera
    from carla.client import VehicleControl
    from carla.planner.planner import Planner
    from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("CARLA")

import logging
import subprocess
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import BoxActionSpace, ImageObservationSpace, StateSpace, \
    VectorObservationSpace
from rl_coach.utils import get_open_port, force_list
from enum import Enum
import os
import signal
from typing import List, Union
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.filters.filter import InputFilter, NoOutputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
import numpy as np


# enum of the available levels and their path
class CarlaLevel(Enum):
    TOWN1 = {"map_name": "Town01", "map_path": "/Game/Maps/Town01"}
    TOWN2 = {"map_name": "Town02", "map_path": "/Game/Maps/Town02"}

key_map = {
    'BRAKE': (274,),  # down arrow
    'GAS': (273,),  # up arrow
    'TURN_LEFT': (276,),  # left arrow
    'TURN_RIGHT': (275,),  # right arrow
    'GAS_AND_TURN_LEFT': (273, 276),
    'GAS_AND_TURN_RIGHT': (273, 275),
    'BRAKE_AND_TURN_LEFT': (274, 276),
    'BRAKE_AND_TURN_RIGHT': (274, 275),
}

CarlaInputFilter = InputFilter(is_a_reference_filter=True)
CarlaInputFilter.add_observation_filter('forward_camera', 'rescaling',
                                        ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([128, 180, 3]),
                                                                                             high=255)))
CarlaInputFilter.add_observation_filter('forward_camera', 'to_grayscale', ObservationRGBToYFilter())
CarlaInputFilter.add_observation_filter('forward_camera', 'to_uint8', ObservationToUInt8Filter(0, 255))
CarlaInputFilter.add_observation_filter('forward_camera', 'stacking', ObservationStackingFilter(4))

CarlaOutputFilter = NoOutputFilter()


class CameraTypes(Enum):
    FRONT = "forward_camera"
    LEFT = "left_camera"
    RIGHT = "right_camera"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    LIDAR = "lidar"


class CarlaEnvironmentParameters(EnvironmentParameters):
    class Quality(Enum):
        LOW = "Low"
        EPIC = "Epic"

    def __init__(self):
        super().__init__()
        self.frame_skip = 3  # the frame skip affects the fps of the server directly. fps = 30 / frameskip
        self.server_height = 512
        self.server_width = 720
        self.camera_height = 128
        self.camera_width = 180
        self.experiment_suite = None  # an optional CARLA experiment suite to use
        self.config = None
        self.level = 'town1'
        self.quality = self.Quality.LOW
        self.cameras = [CameraTypes.FRONT]
        self.weather_id = [1]
        self.verbose = True
        self.episode_max_time = 100000  # miliseconds for each episode
        self.allow_braking = False
        self.separate_actions_for_throttle_and_brake = False
        self.num_speedup_steps = 30
        self.max_speed = 35.0  # km/h
        self.default_input_filter = CarlaInputFilter
        self.default_output_filter = CarlaOutputFilter

    @property
    def path(self):
        return 'rl_coach.environments.carla_environment:CarlaEnvironment'


class CarlaEnvironment(Environment):
    def __init__(self, level: LevelSelection,
                 seed: int, frame_skip: int, human_control: bool, custom_reward_threshold: Union[int, float],
                 visualization_parameters: VisualizationParameters,
                 server_height: int, server_width: int, camera_height: int, camera_width: int,
                 verbose: bool, experiment_suite: ExperimentSuite, config: str, episode_max_time: int,
                 allow_braking: bool, quality: CarlaEnvironmentParameters.Quality,
                 cameras: List[CameraTypes], weather_id: List[int], experiment_path: str,
                 separate_actions_for_throttle_and_brake: bool,
                 num_speedup_steps: int, max_speed: float, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        # server configuration
        self.server_height = server_height
        self.server_width = server_width
        self.port = get_open_port()
        self.host = 'localhost'
        self.map_name = CarlaLevel[level.upper()].value['map_name']
        self.map_path = CarlaLevel[level.upper()].value['map_path']
        self.experiment_path = experiment_path

        # client configuration
        self.verbose = verbose
        self.quality = quality
        self.cameras = cameras
        self.weather_id = weather_id
        self.episode_max_time = episode_max_time
        self.allow_braking = allow_braking
        self.separate_actions_for_throttle_and_brake = separate_actions_for_throttle_and_brake
        self.camera_width = camera_width
        self.camera_height = camera_height

        # setup server settings
        self.experiment_suite = experiment_suite
        self.config = config
        if self.config:
            # load settings from file
            with open(self.config, 'r') as fp:
                self.settings = fp.read()
        else:
            # hard coded settings
            self.settings = CarlaSettings()
            self.settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=False,
                NumberOfVehicles=15,
                NumberOfPedestrians=30,
                WeatherId=random.choice(force_list(self.weather_id)),
                QualityLevel=self.quality.value,
                SeedVehicles=seed,
                SeedPedestrians=seed)
            if seed is None:
                self.settings.randomize_seeds()

            self.settings = self._add_cameras(self.settings, self.cameras, self.camera_width, self.camera_height)

        # open the server
        self.server = self._open_server()

        logging.disable(40)

        # open the client
        self.game = CarlaClient(self.host, self.port, timeout=99999999)
        self.game.connect()
        if self.experiment_suite:
            self.current_experiment_idx = 0
            self.current_experiment = self.experiment_suite.get_experiments()[self.current_experiment_idx]
            self.scene = self.game.load_settings(self.current_experiment.conditions)
        else:
            self.scene = self.game.load_settings(self.settings)

        # get available start positions
        self.positions = self.scene.player_start_spots
        self.num_positions = len(self.positions)
        self.current_start_position_idx = 0
        self.current_pose = 0

        # state space
        self.state_space = StateSpace({
            "measurements": VectorObservationSpace(4, measurements_names=["forward_speed", "x", "y", "z"])
        })
        for camera in self.scene.sensors:
            self.state_space[camera.name] = ImageObservationSpace(
                shape=np.array([self.camera_height, self.camera_width, 3]),
                high=255)

        # action space
        if self.separate_actions_for_throttle_and_brake:
            self.action_space = BoxActionSpace(shape=3, low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]),
                                               descriptions=["steer", "gas", "brake"])
        else:
            self.action_space = BoxActionSpace(shape=2, low=np.array([-1, -1]), high=np.array([1, 1]),
                                               descriptions=["steer", "gas_and_brake"])

        # human control
        if self.human_control:
            # convert continuous action space to discrete
            self.steering_strength = 0.5
            self.gas_strength = 1.0
            self.brake_strength = 0.5
            # TODO: reverse order of actions
            self.action_space = PartialDiscreteActionSpaceMap(
                target_actions=[[0., 0.],
                                [0., -self.steering_strength],
                                [0., self.steering_strength],
                                [self.gas_strength, 0.],
                                [-self.brake_strength, 0],
                                [self.gas_strength, -self.steering_strength],
                                [self.gas_strength, self.steering_strength],
                                [self.brake_strength, -self.steering_strength],
                                [self.brake_strength, self.steering_strength]],
                descriptions=['NO-OP', 'TURN_LEFT', 'TURN_RIGHT', 'GAS', 'BRAKE',
                              'GAS_AND_TURN_LEFT', 'GAS_AND_TURN_RIGHT',
                              'BRAKE_AND_TURN_LEFT', 'BRAKE_AND_TURN_RIGHT']
            )

            # map keyboard keys to actions
            for idx, action in enumerate(self.action_space.descriptions):
                for key in key_map.keys():
                    if action == key:
                        self.key_to_action[key_map[key]] = idx

        self.num_speedup_steps = num_speedup_steps
        self.max_speed = max_speed

        # measurements
        self.autopilot = None
        self.planner = Planner(self.map_name)

        # env initialization
        self.reset_internal_state(True)

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            self.renderer.create_screen(image.shape[1], image.shape[0])

    def _add_cameras(self, settings, cameras, camera_width, camera_height):
        # add a front facing camera
        if CameraTypes.FRONT in cameras:
            camera = Camera(CameraTypes.FRONT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, 0, 0)
            settings.add_sensor(camera)

        # add a left facing camera
        if CameraTypes.LEFT in cameras:
            camera = Camera(CameraTypes.LEFT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, -30, 0)
            settings.add_sensor(camera)

        # add a right facing camera
        if CameraTypes.RIGHT in cameras:
            camera = Camera(CameraTypes.RIGHT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, 30, 0)
            settings.add_sensor(camera)

        # add a front facing depth camera
        if CameraTypes.DEPTH in cameras:
            camera = Camera(CameraTypes.DEPTH.value)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(0.2, 0, 1.3)
            camera.set_rotation(8, 30, 0)
            camera.PostProcessing = 'Depth'
            settings.add_sensor(camera)

        # add a front facing semantic segmentation camera
        if CameraTypes.SEGMENTATION in cameras:
            camera = Camera(CameraTypes.SEGMENTATION.value)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(0.2, 0, 1.3)
            camera.set_rotation(8, 30, 0)
            camera.PostProcessing = 'SemanticSegmentation'
            settings.add_sensor(camera)

        return settings

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self.planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _open_server(self):
        log_path = path.join(self.experiment_path if self.experiment_path is not None else '.', 'logs',
                             "CARLA_LOG_{}.txt".format(self.port))
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        with open(log_path, "wb") as out:
            cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map_path,
                   "-benchmark", "-carla-server", "-fps={}".format(30 / self.frame_skip),
                   "-world-port={}".format(self.port),
                   "-windowed -ResX={} -ResY={}".format(self.server_width, self.server_height),
                   "-carla-no-hud"]

            if self.config:
                cmd.append("-carla-settings={}".format(self.config))
            p = subprocess.Popen(cmd, stdout=out, stderr=out)

        return p

    def _close_server(self):
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)

    def _update_state(self):
        # get measurements and observations
        measurements = []
        while type(measurements) == list:
            measurements, sensor_data = self.game.read_data()
        self.state = {}

        for camera in self.scene.sensors:
            self.state[camera.name] = sensor_data[camera.name].data

        self.location = [measurements.player_measurements.transform.location.x,
                         measurements.player_measurements.transform.location.y,
                         measurements.player_measurements.transform.location.z]

        self.distance_from_goal = np.linalg.norm(np.array(self.location[:2]) -
                                                 [self.current_goal.location.x, self.current_goal.location.y])

        is_collision = measurements.player_measurements.collision_vehicles != 0 \
                       or measurements.player_measurements.collision_pedestrians != 0 \
                       or measurements.player_measurements.collision_other != 0

        speed_reward = measurements.player_measurements.forward_speed - 1
        if speed_reward > 30.:
            speed_reward = 30.
        self.reward = speed_reward \
                      - (measurements.player_measurements.intersection_otherlane * 5) \
                      - (measurements.player_measurements.intersection_offroad * 5) \
                      - is_collision * 100 \
                      - np.abs(self.control.steer) * 10

        # update measurements
        self.measurements = [measurements.player_measurements.forward_speed] + self.location
        self.autopilot = measurements.player_measurements.autopilot_control

        # The directions to reach the goal (0 Follow lane, 1 Left, 2 Right, 3 Straight)
        directions = int(self._get_directions(measurements.player_measurements.transform, self.current_goal) - 2)
        self.state['high_level_command'] = directions

        if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
            self.done = True

        self.state['measurements'] = np.array(self.measurements)

    def _take_action(self, action):
        self.control = VehicleControl()

        if self.separate_actions_for_throttle_and_brake:
            self.control.steer = np.clip(action[0], -1, 1)
            self.control.throttle = np.clip(action[1], 0, 1)
            self.control.brake = np.clip(action[2], 0, 1)
        else:
            # transform the 2 value action (steer, throttle - brake) into a 3 value action (steer, throttle, brake)
            self.control.steer = np.clip(action[0], -1, 1)
            self.control.throttle = np.clip(action[1], 0, 1)
            self.control.brake = np.abs(np.clip(action[1], -1, 0))

        # prevent braking
        if not self.allow_braking or self.control.brake < 0.1 or self.control.throttle > self.control.brake:
            self.control.brake = 0

        # prevent over speeding
        if hasattr(self, 'measurements') and self.measurements[0] * 3.6 > self.max_speed and self.control.brake == 0:
            self.control.throttle = 0.0

        self.control.hand_brake = False
        self.control.reverse = False

        self.game.send_control(self.control)

    def _load_experiment(self, experiment_idx):
        self.current_experiment = self.experiment_suite.get_experiments()[experiment_idx]
        self.scene = self.game.load_settings(self.current_experiment.conditions)
        self.positions = self.scene.player_start_spots
        self.num_positions = len(self.positions)
        self.current_start_position_idx = 0
        self.current_pose = 0

    def _restart_environment_episode(self, force_environment_reset=False):
        # select start and end positions
        if self.experiment_suite:
            # if an expeirent suite is available, follow its given poses
            if self.current_pose >= len(self.current_experiment.poses):
                # load a new experiment
                self.current_experiment_idx = (self.current_experiment_idx + 1) % len(self.experiment_suite.get_experiments())
                self._load_experiment(self.current_experiment_idx)

            self.current_start_position_idx = self.current_experiment.poses[self.current_pose][0]
            self.current_goal = self.positions[self.current_experiment.poses[self.current_pose][1]]
            self.current_pose += 1
        else:
            # go over all the possible positions in a cyclic manner
            self.current_start_position_idx = (self.current_start_position_idx + 1) % self.num_positions

            # choose a random goal destination
            self.current_goal = random.choice(self.positions)

        try:
            self.game.start_episode(self.current_start_position_idx)
        except:
            self.game.connect()
            self.game.start_episode(self.current_start_position_idx)

        # start the game with some initial speed
        for i in range(self.num_speedup_steps):
            self.control = VehicleControl(throttle=1.0, brake=0, steer=0, hand_brake=False, reverse=False)
            self.game.send_control(VehicleControl())

    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        image = [self.state[camera.name] for camera in self.scene.sensors]
        image = np.vstack(image)
        return image
