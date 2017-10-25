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

import sys
import gym
import numpy as np
try:
    import roboschool
    from OpenGL import GL
except ImportError:
    from logger import failed_imports
    failed_imports.append("RoboSchool")

try:
    from gym_extensions.continuous import mujoco
except:
    from logger import failed_imports
    failed_imports.append("GymExtensions")

try:
    import pybullet_envs
except ImportError:
    from logger import failed_imports
    failed_imports.append("PyBullet")

from gym import wrappers
from utils import force_list, RunPhase
from environments.environment_wrapper import EnvironmentWrapper

i = 0


class GymEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, tuning_parameters):
        EnvironmentWrapper.__init__(self, tuning_parameters)

        # env parameters
        self.env = gym.make(self.env_id)
        if self.seed is not None:
            self.env.seed(self.seed)

        # self.env_spec = gym.spec(self.env_id)
        self.discrete_controls = type(self.env.action_space) != gym.spaces.box.Box

        # pybullet requires rendering before resetting the environment, but other gym environments (Pendulum) will crash
        try:
            if self.is_rendered:
                self.render()
        except:
            pass

        o = self.reset(True)['observation']

        # render
        if self.is_rendered:
            self.render()

        self.is_state_type_image = len(o.shape) > 1
        if self.is_state_type_image:
            self.width = o.shape[1]
            self.height = o.shape[0]
        else:
            self.width = o.shape[0]

        self.actions_description = {}
        if self.discrete_controls:
            self.action_space_size = self.env.action_space.n
            self.action_space_abs_range = 0
        else:
            self.action_space_size = self.env.action_space.shape[0]
            self.action_space_high = self.env.action_space.high
            self.action_space_low = self.env.action_space.low
            self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
        self.actions = {i: i for i in range(self.action_space_size)}
        self.timestep_limit = self.env.spec.timestep_limit
        self.current_ale_lives = 0
        self.measurements_size = len(self.step(0)['info'].keys())

        # env intialization
        self.observation = o
        self.reward = 0
        self.done = False
        self.last_action = self.actions[0]

    def render(self):
        self.env.render()

    def step(self, action_idx):

        if action_idx is None:
            action_idx = self.last_action_idx

        self.last_action_idx = action_idx

        if self.discrete_controls:
            action = self.actions[action_idx]
        else:
            action = action_idx

        if hasattr(self.env.env, 'ale'):
            prev_ale_lives = self.env.env.ale.lives()

        # pendulum-v0 for example expects a list
        if not self.discrete_controls:
            # catching cases where the action for continuous control is a number instead of a list the
            # size of the action space
            if type(action_idx) == int and action_idx == 0:
                # deal with the "reset" action 0
                action = [0] * self.env.action_space.shape[0]
            action = np.array(force_list(action))
            # removing redundant dimensions such that the action size will match the expected action size from gym
            if action.shape != self.env.action_space.shape:
                action = np.squeeze(action)
            action = np.clip(action, self.action_space_low, self.action_space_high)

        self.observation, self.reward, self.done, self.info = self.env.step(action)

        if hasattr(self.env.env, 'ale') and self.phase == RunPhase.TRAIN:
            # signal termination for breakout life loss
            if prev_ale_lives != self.env.env.ale.lives():
                self.done = True

        if any(env in self.env_id for env in ["Breakout", "Pong"]):
            # crop image
            self.observation = self.observation[34:195, :, :]

        if self.is_rendered:
            self.render()

        return {'observation': self.observation,
                'reward': self.reward,
                'done': self.done,
                'action': self.last_action_idx,
                'info': self.info}

    def _restart_environment_episode(self, force_environment_reset=False):
        # prevent reset of environment if there are ale lives left
        if "Breakout" in self.env_id and self.env.env.ale.lives() > 0 and not force_environment_reset:
            return self.observation

        if self.seed:
            self.env.seed(self.seed)
        observation = self.env.reset()
        while observation is None:
            observation = self.step(0)['observation']

        if "Breakout" in self.env_id:
            # crop image
            observation = observation[34:195, :, :]

        self.observation = observation

        return observation

    def get_rendered_image(self):
        return self.env.render(mode='rgb_array')
