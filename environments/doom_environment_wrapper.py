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
    from logger import failed_imports
    failed_imports.append("ViZDoom")

import numpy as np
from environments.environment_wrapper import EnvironmentWrapper
from os import path, environ
from utils import *
from logger import *


# enum of the available levels and their path
class DoomLevel(Enum):
    BASIC = "basic.cfg"
    DEFEND = "defend_the_center.cfg"
    DEATHMATCH = "deathmatch.cfg"
    MY_WAY_HOME = "my_way_home.cfg"
    TAKE_COVER = "take_cover.cfg"
    HEALTH_GATHERING = "health_gathering.cfg"
    HEALTH_GATHERING_SUPREME = "health_gathering_supreme.cfg"
    DEFEND_THE_LINE = "defend_the_line.cfg"
    DEADLY_CORRIDOR = "deadly_corridor.cfg"

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


class DoomEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, tuning_parameters):
        EnvironmentWrapper.__init__(self, tuning_parameters)

        # load the emulator with the required level
        self.level = DoomLevel().get(self.tp.env.level)
        self.scenarios_dir = path.join(environ.get('VIZDOOM_ROOT'), 'scenarios')
        self.game = vizdoom.DoomGame()
        self.game.load_config(path.join(self.scenarios_dir, self.level))
        self.game.set_window_visible(False)
        self.game.add_game_args("+vid_forcesurface 1")

        self.wait_for_explicit_human_action = True
        if self.human_control:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
            self.renderer.create_screen(640, 480)
        elif self.is_rendered:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
            self.renderer.create_screen(320, 240)
        else:
            # lower resolution since we actually take only 76x60 and we don't need to render
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)

        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.init()

        # action space
        self.action_space_abs_range = 0
        self.actions = {}
        self.action_space_size = self.game.get_available_buttons_size() + 1
        self.action_vector_size = self.action_space_size - 1
        self.actions[0] = [0] * self.action_vector_size
        for action_idx in range(self.action_vector_size):
            self.actions[action_idx + 1] = [0] * self.action_vector_size
            self.actions[action_idx + 1][action_idx] = 1
        self.actions_description = ['NO-OP']
        self.actions_description += [str(action).split(".")[1] for action in self.game.get_available_buttons()]
        for idx, action in enumerate(self.actions_description):
            if action in key_map.keys():
                self.key_to_action[(key_map[action],)] = idx

        # measurement
        self.measurements_size = self.game.get_state().game_variables.shape

        self.width = self.game.get_screen_width()
        self.height = self.game.get_screen_height()
        if self.tp.seed is not None:
            self.game.set_seed(self.tp.seed)
        self.reset()

    def _update_state(self):
        # extract all data from the current state
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            self.state = {
                'observation': state.screen_buffer,
                'measurements': state.game_variables,
            }
        self.reward = self.game.get_last_reward()
        self.done = self.game.is_episode_finished()

    def _take_action(self, action_idx):
        self.game.make_action(self._idx_to_action(action_idx), self.frame_skip)

    def _preprocess_observation(self, observation):
        if observation is None:
            return None

        # for the last step we get no new observation, so we shouldn't preprocess it
        if self.done:
            return observation

        # move the channel to the last axis
        observation = np.transpose(observation, (1, 2, 0))
        return observation

    def _restart_environment_episode(self, force_environment_reset=False):
        self.game.new_episode()
