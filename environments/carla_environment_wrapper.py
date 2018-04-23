import sys
from os import path, environ

try:
    if 'CARLA_ROOT' in environ:
        sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
    from carla.client import CarlaClient
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.sensor import Camera
    from carla.client import VehicleControl
except ImportError:
    from logger import failed_imports
    failed_imports.append("CARLA")

import numpy as np
import time
import logging
import subprocess
import signal
from environments.environment_wrapper import EnvironmentWrapper
from utils import *
from logger import screen, logger
from PIL import Image


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


class CarlaEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, tuning_parameters):
        EnvironmentWrapper.__init__(self, tuning_parameters)

        self.tp = tuning_parameters

        # server configuration
        self.server_height = self.tp.env.server_height
        self.server_width = self.tp.env.server_width
        self.port = get_open_port()
        self.host = 'localhost'
        self.map = CarlaLevel().get(self.tp.env.level)

        # client configuration
        self.verbose = self.tp.env.verbose
        self.depth = self.tp.env.depth
        self.stereo = self.tp.env.stereo
        self.semantic_segmentation = self.tp.env.semantic_segmentation
        self.height = self.server_height * (1 + int(self.depth) + int(self.semantic_segmentation))
        self.width = self.server_width * (1 + int(self.stereo))
        self.size = (self.width, self.height)

        self.config = self.tp.env.config
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
                WeatherId=1)
            self.settings.randomize_seeds()

            # add cameras
            camera = Camera('CameraRGB')
            camera.set_image_size(self.width, self.height)
            camera.set_position(200, 0, 140)
            camera.set_rotation(0, 0, 0)
            self.settings.add_sensor(camera)

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
        self.discrete_controls = False
        self.action_space_size = 2
        self.action_space_high = [1, 1]
        self.action_space_low = [-1, -1]
        self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
        self.steering_strength = 0.5
        self.gas_strength = 1.0
        self.brake_strength = 0.5
        self.actions = {0: [0., 0.],
                        1: [0., -self.steering_strength],
                        2: [0., self.steering_strength],
                        3: [self.gas_strength, 0.],
                        4: [-self.brake_strength, 0],
                        5: [self.gas_strength, -self.steering_strength],
                        6: [self.gas_strength, self.steering_strength],
                        7: [self.brake_strength, -self.steering_strength],
                        8: [self.brake_strength, self.steering_strength]}
        self.actions_description = ['NO-OP', 'TURN_LEFT', 'TURN_RIGHT', 'GAS', 'BRAKE',
                                    'GAS_AND_TURN_LEFT', 'GAS_AND_TURN_RIGHT',
                                    'BRAKE_AND_TURN_LEFT', 'BRAKE_AND_TURN_RIGHT']
        for idx, action in enumerate(self.actions_description):
            for key in key_map.keys():
                if action == key:
                    self.key_to_action[key_map[key]] = idx
        self.num_speedup_steps = 30

        # measurements
        self.measurements_size = (1,)
        self.autopilot = None

        # env initialization
        self.reset(True)

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            self.renderer.create_screen(image.shape[1], image.shape[0])

    def _open_server(self):
        log_path = path.join(logger.experiments_path, "CARLA_LOG_{}.txt".format(self.port))
        with open(log_path, "wb") as out:
            cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map,
                                  "-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(self.port),
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

        self.location = (measurements.player_measurements.transform.location.x,
                         measurements.player_measurements.transform.location.y,
                         measurements.player_measurements.transform.location.z)

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
        self.state = {
            'observation': sensor_data['CameraRGB'].data,
            'measurements': [measurements.player_measurements.forward_speed],
        }
        self.autopilot = measurements.player_measurements.autopilot_control

        # action_p = ['%.2f' % member for member in [self.control.throttle, self.control.steer]]
        # screen.success('REWARD: %.2f, ACTIONS: %s' % (self.reward, action_p))

        if (measurements.game_timestamp >= self.tp.env.episode_max_time) or is_collision:
            # screen.success('EPISODE IS DONE. GameTime: {}, Collision: {}'.format(str(measurements.game_timestamp),
            #                                                                      str(is_collision)))
            self.done = True

    def _take_action(self, action_idx):
        if type(action_idx) == int:
            action = self.actions[action_idx]
        else:
            action = action_idx
        self.last_action_idx = action

        self.control = VehicleControl()
        self.control.throttle = np.clip(action[0], 0, 1)
        self.control.steer = np.clip(action[1], -1, 1)
        self.control.brake = np.abs(np.clip(action[0], -1, 0))
        if not self.tp.env.allow_braking:
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
        state = None
        for i in range(self.num_speedup_steps):
            state = self.step([1.0, 0])['state']
        self.state = state

        return state
