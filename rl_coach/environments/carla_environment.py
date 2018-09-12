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
    TOWN1 = "/Game/Maps/Town01"
    TOWN2 = "/Game/Maps/Town02"

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
        self.config = None #'environments/CarlaSettings.ini'  # TODO: remove the config to prevent confusion
        self.level = 'town1'
        self.quality = self.Quality.LOW
        self.cameras = [CameraTypes.FRONT]
        self.weather_id = [1]
        self.verbose = True
        self.episode_max_time = 100000  # miliseconds for each episode
        self.allow_braking = False
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
                 verbose: bool, config: str, episode_max_time: int,
                 allow_braking: bool, quality: CarlaEnvironmentParameters.Quality,
                 cameras: List[CameraTypes], weather_id: List[int], experiment_path: str, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        # server configuration
        self.server_height = server_height
        self.server_width = server_width
        self.port = get_open_port()
        self.host = 'localhost'
        self.map = self.env_id
        self.experiment_path = experiment_path

        # client configuration
        self.verbose = verbose
        self.quality = quality
        self.cameras = cameras
        self.weather_id = weather_id
        self.episode_max_time = episode_max_time
        self.allow_braking = allow_braking
        self.camera_width = camera_width
        self.camera_height = camera_height

        # state space
        self.state_space = StateSpace({
            "measurements": VectorObservationSpace(4, measurements_names=["forward_speed", "x", "y", "z"])
        })
        for camera in self.cameras:
            self.state_space[camera.value] = ImageObservationSpace(
                shape=np.array([self.camera_height, self.camera_width, 3]),
                high=255)

        # setup server settings
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
        scene = self.game.load_settings(self.settings)

        # get available start positions
        positions = scene.player_start_spots
        self.num_pos = len(positions)
        self.iterator_start_positions = 0

        # action space
        self.action_space = BoxActionSpace(shape=2, low=np.array([-1, -1]), high=np.array([1, 1]))

        # human control
        if self.human_control:
            # convert continuous action space to discrete
            self.steering_strength = 0.5
            self.gas_strength = 1.0
            self.brake_strength = 0.5
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

        self.num_speedup_steps = 30

        # measurements
        self.autopilot = None

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

    def _open_server(self):
        log_path = path.join(self.experiment_path if self.experiment_path is not None else '.', 'logs',
                             "CARLA_LOG_{}.txt".format(self.port))
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        with open(log_path, "wb") as out:
            cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map,
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

        for camera in self.cameras:
            self.state[camera.value] = sensor_data[camera.value].data

        self.location = [measurements.player_measurements.transform.location.x,
                         measurements.player_measurements.transform.location.y,
                         measurements.player_measurements.transform.location.z]

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

        # action_p = ['%.2f' % member for member in [self.control.throttle, self.control.steer]]
        # screen.success('REWARD: %.2f, ACTIONS: %s' % (self.reward, action_p))

        if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
            # screen.success('EPISODE IS DONE. GameTime: {}, Collision: {}'.format(str(measurements.game_timestamp),
            #                                                                      str(is_collision)))
            self.done = True

        self.state['measurements'] = np.array(self.measurements)

    def _take_action(self, action):
        self.control = VehicleControl()
        self.control.throttle = np.clip(action[0], 0, 1)
        self.control.steer = np.clip(action[1], -1, 1)
        self.control.brake = np.abs(np.clip(action[0], -1, 0))
        if not self.allow_braking:
            self.control.brake = 0
        self.control.hand_brake = False
        self.control.reverse = False

        self.game.send_control(self.control)

    def _restart_environment_episode(self, force_environment_reset=False):
        self.iterator_start_positions += 1
        if self.iterator_start_positions >= self.num_pos:
            self.iterator_start_positions = 0

        try:
            self.game.start_episode(self.iterator_start_positions)
        except:
            self.game.connect()
            self.game.start_episode(self.iterator_start_positions)

        # start the game with some initial speed
        for i in range(self.num_speedup_steps):
            self._take_action([1.0, 0])

    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        image = [self.state[camera.value] for camera in self.cameras]
        image = np.vstack(image)
        return image
