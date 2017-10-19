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


class DoomEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, tuning_parameters):
        EnvironmentWrapper.__init__(self, tuning_parameters)

        # load the emulator with the required level
        self.level = DoomLevel().get(self.tp.env.level)
        self.scenarios_dir = path.join(environ.get('VIZDOOM_ROOT'), 'scenarios')
        self.game = vizdoom.DoomGame()
        self.game.load_config(path.join(self.scenarios_dir, self.level))
        self.game.set_window_visible(self.is_rendered)
        self.game.add_game_args("+vid_forcesurface 1")
        if self.is_rendered:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
        else:
            # lower resolution since we actually take only 76x60 and we don't need to render
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.init()

        self.action_space_abs_range = 0
        self.actions = {}
        self.action_space_size = self.game.get_available_buttons_size()
        for action_idx in range(self.action_space_size):
            self.actions[action_idx] = [0] * self.action_space_size
            self.actions[action_idx][action_idx] = 1
        self.actions_description = [str(action) for action in self.game.get_available_buttons()]
        self.measurements_size = self.game.get_state().game_variables.shape

        self.width = self.game.get_screen_width()
        self.height = self.game.get_screen_height()
        if self.tp.seed is not None:
            self.game.set_seed(self.tp.seed)
        self.reset()

    def _update_observation_and_measurements(self):
        # extract all data from the current state
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            self.observation = self._preprocess_observation(state.screen_buffer)
            self.measurements = state.game_variables
        self.done = self.game.is_episode_finished()

    def step(self, action_idx):
        self.reward = 0
        for frame in range(self.tp.env.frame_skip):
            self.reward += self.game.make_action(self._idx_to_action(action_idx))
            self._update_observation_and_measurements()
            if self.done:
                break

        return {'observation': self.observation,
                'reward': self.reward,
                'done': self.done,
                'action': action_idx,
                'measurements': self.measurements}

    def _preprocess_observation(self, observation):
        if observation is None:
            return None
        # move the channel to the last axis
        observation = np.transpose(observation, (1, 2, 0))
        return observation

    def _restart_environment_episode(self, force_environment_reset=False):
        self.game.new_episode()
