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

try:
    import vizdoom
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("ViZDoom")

import os
from enum import Enum
from os import path, environ
from typing import Union, List

import numpy as np

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.filters.action.full_discrete_action_space_map import FullDiscreteActionSpaceMap
from rl_coach.filters.filter import InputFilter, OutputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.spaces import MultiSelectActionSpace, ImageObservationSpace, \
    VectorObservationSpace, StateSpace


# enum of the available levels and their path
class DoomLevel(Enum):
    BASIC = "basic.cfg"
    DEFEND = "defend_the_center.cfg"
    DEATHMATCH = "deathmatch.cfg"
    MY_WAY_HOME = "my_way_home.cfg"
    TAKE_COVER = "take_cover.cfg"
    HEALTH_GATHERING = "health_gathering.cfg"
    HEALTH_GATHERING_SUPREME_COACH_LOCAL = "D2_navigation.cfg"  # from https://github.com/IntelVCL/DirectFuturePrediction/tree/master/maps
    DEFEND_THE_LINE = "defend_the_line.cfg"
    DEADLY_CORRIDOR = "deadly_corridor.cfg"
    BATTLE_COACH_LOCAL = "D3_battle.cfg"  # from https://github.com/IntelVCL/DirectFuturePrediction/tree/master/maps

key_map = {
    'NO-OP': 96,  # `
    'ATTACK': 13,  # enter
    'CROUCH': 306,  # ctrl
    'DROP_SELECTED_ITEM': ord("t"),
    'DROP_SELECTED_WEAPON': ord("t"),
    'JUMP': 32,  # spacebar
    'LAND': ord("l"),
    'LOOK_DOWN': 274,  # down arrow
    'LOOK_UP': 273,  # up arrow
    'MOVE_BACKWARD': ord("s"),
    'MOVE_DOWN': ord("s"),
    'MOVE_FORWARD': ord("w"),
    'MOVE_LEFT': 276,
    'MOVE_RIGHT': 275,
    'MOVE_UP': ord("w"),
    'RELOAD': ord("r"),
    'SELECT_NEXT_WEAPON': ord("q"),
    'SELECT_PREV_WEAPON': ord("e"),
    'SELECT_WEAPON0': ord("0"),
    'SELECT_WEAPON1': ord("1"),
    'SELECT_WEAPON2': ord("2"),
    'SELECT_WEAPON3': ord("3"),
    'SELECT_WEAPON4': ord("4"),
    'SELECT_WEAPON5': ord("5"),
    'SELECT_WEAPON6': ord("6"),
    'SELECT_WEAPON7': ord("7"),
    'SELECT_WEAPON8': ord("8"),
    'SELECT_WEAPON9': ord("9"),
    'SPEED': 304,  # shift
    'STRAFE': 9,  # tab
    'TURN180': ord("u"),
    'TURN_LEFT': ord("a"),  # left arrow
    'TURN_RIGHT': ord("d"),  # right arrow
    'USE': ord("f"),
}


DoomInputFilter = InputFilter(is_a_reference_filter=True)
DoomInputFilter.add_observation_filter('observation', 'rescaling',
                                       ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([60, 76, 3]),
                                                                                            high=255)))
DoomInputFilter.add_observation_filter('observation', 'to_grayscale', ObservationRGBToYFilter())
DoomInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
DoomInputFilter.add_observation_filter('observation', 'stacking', ObservationStackingFilter(3))


DoomOutputFilter = OutputFilter(is_a_reference_filter=True)
DoomOutputFilter.add_action_filter('to_discrete', FullDiscreteActionSpaceMap())


class DoomEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = DoomInputFilter
        self.default_output_filter = DoomOutputFilter
        self.cameras = [DoomEnvironment.CameraTypes.OBSERVATION]

    @property
    def path(self):
        return 'rl_coach.environments.doom_environment:DoomEnvironment'


class DoomEnvironment(Environment):
    class CameraTypes(Enum):
        OBSERVATION = ("observation", "screen_buffer")
        DEPTH = ("depth", "depth_buffer")
        LABELS = ("labels", "labels_buffer")
        MAP = ("map", "automap_buffer")

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 cameras: List[CameraTypes], **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        self.cameras = cameras

        # load the emulator with the required level
        self.level = DoomLevel[level.upper()]
        local_scenarios_path = path.join(os.path.dirname(os.path.realpath(__file__)), 'doom')
        self.scenarios_dir = local_scenarios_path if 'COACH_LOCAL' in level \
            else path.join(environ.get('VIZDOOM_ROOT'), 'scenarios')

        self.game = vizdoom.DoomGame()
        self.game.load_config(path.join(self.scenarios_dir, self.level.value))
        self.game.set_window_visible(False)
        self.game.add_game_args("+vid_forcesurface 1")

        self.wait_for_explicit_human_action = True
        if self.human_control:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        elif self.is_rendered:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
        else:
            # lower resolution since we actually take only 76x60 and we don't need to render
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)

        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        for camera in self.cameras:
            if hasattr(self.game, 'set_{}_enabled'.format(camera.value[1])):
                getattr(self.game, 'set_{}_enabled'.format(camera.value[1]))(True)
        self.game.init()

        # actions
        actions_description = ['NO-OP']
        actions_description += [str(action).split(".")[1] for action in self.game.get_available_buttons()]
        actions_description = actions_description[::-1]
        self.action_space = MultiSelectActionSpace(self.game.get_available_buttons_size(),
                                                   max_simultaneous_selected_actions=1,
                                                   descriptions=actions_description,
                                                   allow_no_action_to_be_selected=True)

        # human control
        if self.human_control:
            # TODO: add this to the action space
            # map keyboard keys to actions
            for idx, action in enumerate(self.action_space.descriptions):
                if action in key_map.keys():
                    self.key_to_action[(key_map[action],)] = idx

        # states
        self.state_space = StateSpace({
            "measurements": VectorObservationSpace(self.game.get_state().game_variables.shape[0],
                                                   measurements_names=[str(m) for m in
                                                                       self.game.get_available_game_variables()])
        })
        for camera in self.cameras:
            self.state_space[camera.value[0]] = ImageObservationSpace(
                shape=np.array([self.game.get_screen_height(), self.game.get_screen_width(), 3]),
                high=255)

        # seed
        if seed is not None:
            self.game.set_seed(seed)
        self.reset_internal_state()

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            self.renderer.create_screen(image.shape[1], image.shape[0])

    def _update_state(self):
        # extract all data from the current state
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            self.measurements = state.game_variables
            self.state = {'measurements': self.measurements}
            for camera in self.cameras:
                observation = getattr(state, camera.value[1])
                if len(observation.shape) == 3:
                    self.state[camera.value[0]] = np.transpose(observation, (1, 2, 0))
                elif len(observation.shape) == 2:
                    self.state[camera.value[0]] = np.repeat(np.expand_dims(observation, -1), 3, axis=-1)

        self.reward = self.game.get_last_reward()
        self.done = self.game.is_episode_finished()

    def _take_action(self, action):
        self.game.make_action(list(action), self.frame_skip)

    def _restart_environment_episode(self, force_environment_reset=False):
        self.game.new_episode()

    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        image = [self.state[camera.value[0]] for camera in self.cameras]
        image = np.vstack(image)
        return image
