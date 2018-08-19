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

import os

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen


class PendulumWithGoals(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30
    }

    def __init__(self, goal_reaching_thresholds=np.array([0.075, 0.075, 0.75]),
                 goal_not_reached_penalty=-1, goal_reached_reward=0, terminate_on_goal_reaching=True,
                 time_limit=1000, frameskip=1, random_goals_instead_of_standing_goal=False,
                 polar_coordinates: bool=False):
        super().__init__()
        dir = os.path.dirname(__file__)
        model = load_model_from_path(dir + "/pendulum_with_goals.xml")

        self.sim = MjSim(model)
        self.viewer = None
        self.rgb_viewer = None

        self.frameskip = frameskip
        self.goal = None
        self.goal_reaching_thresholds = goal_reaching_thresholds
        self.goal_not_reached_penalty = goal_not_reached_penalty
        self.goal_reached_reward = goal_reached_reward
        self.terminate_on_goal_reaching = terminate_on_goal_reaching
        self.time_limit = time_limit
        self.current_episode_steps_counter = 0
        self.random_goals_instead_of_standing_goal = random_goals_instead_of_standing_goal
        self.polar_coordinates = polar_coordinates

        # spaces definition
        self.action_space = spaces.Box(low=-self.sim.model.actuator_ctrlrange[:, 1],
                                       high=self.sim.model.actuator_ctrlrange[:, 1],
                                       dtype=np.float32)
        if self.polar_coordinates:
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=np.array([-np.pi, -15]),
                                          high=np.array([np.pi, 15]),
                                          dtype=np.float32),
                "desired_goal": spaces.Box(low=np.array([-np.pi, -15]),
                                           high=np.array([np.pi, 15]),
                                           dtype=np.float32),
                "achieved_goal": spaces.Box(low=np.array([-np.pi, -15]),
                                            high=np.array([np.pi, 15]),
                                            dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=np.array([-1, -1, -15]),
                                          high=np.array([1, 1, 15]),
                                          dtype=np.float32),
                "desired_goal": spaces.Box(low=np.array([-1, -1, -15]),
                                           high=np.array([1, 1, 15]),
                                           dtype=np.float32),
                "achieved_goal": spaces.Box(low=np.array([-1, -1, -15]),
                                            high=np.array([1, 1, 15]),
                                            dtype=np.float32)
            })

        self.spec = EnvSpec('PendulumWithGoals-v0')
        self.spec.reward_threshold = self.goal_not_reached_penalty * self.time_limit

        self.reset()

    def _goal_reached(self):
        observation = self._get_obs()
        if np.any(np.abs(observation['achieved_goal'] - observation['desired_goal']) > self.goal_reaching_thresholds):
            return False
        else:
            return True

    def _terminate(self):
        if (self._goal_reached() and self.terminate_on_goal_reaching) or \
                        self.current_episode_steps_counter >= self.time_limit:
            return True
        else:
            return False

    def _reward(self):
        if self._goal_reached():
            return self.goal_reached_reward
        else:
            return self.goal_not_reached_penalty

    def step(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(self.frameskip):
            self.sim.step()

        self.current_episode_steps_counter += 1

        state = self._get_obs()

        # visualize the angular velocities
        state_velocity = np.copy(state['observation'][-1] / 20)
        goal_velocity = self.goal[-1] / 20
        self.sim.model.site_size[2] = np.array([0.01, 0.01, state_velocity])
        self.sim.data.mocap_pos[2] = np.array([0.85, 0, 0.75 + state_velocity])
        self.sim.model.site_size[3] = np.array([0.01, 0.01, goal_velocity])
        self.sim.data.mocap_pos[3] = np.array([1.15, 0, 0.75 + goal_velocity])

        return state, self._reward(), self._terminate(), {}

    def _get_obs(self):

        """
        y

        ^
        |____
        |   /
        |  /
        |~/
        |/
        --------> x

        """

        # observation
        angle = self.sim.data.qpos
        angular_velocity = self.sim.data.qvel
        if self.polar_coordinates:
            observation = np.concatenate([angle - np.pi, angular_velocity])
        else:
            x = np.sin(angle)
            y = np.cos(angle)  # qpos is the angle relative to a standing pole
            observation = np.concatenate([x, y, angular_velocity])

        return {
            "observation": observation,
            "desired_goal": self.goal,
            "achieved_goal": observation
        }

    def reset(self):
        self.current_episode_steps_counter = 0

        # set initial state
        angle = np.random.uniform(np.pi / 4, 7 * np.pi / 4)
        angular_velocity = np.random.uniform(-0.05, 0.05)
        self.sim.data.qpos[0] = angle
        self.sim.data.qvel[0] = angular_velocity
        self.sim.step()

        # goal
        if self.random_goals_instead_of_standing_goal:
            angle_target = np.random.uniform(-np.pi / 8, np.pi / 8)
            angular_velocity_target = np.random.uniform(-0.2, 0.2)
        else:
            angle_target = 0
            angular_velocity_target = 0

        # convert target values to goal
        x_target = np.sin(angle_target)
        y_target = np.cos(angle_target)
        if self.polar_coordinates:
            self.goal = np.array([angle_target - np.pi, angular_velocity_target])
        else:
            self.goal = np.array([x_target, y_target, angular_velocity_target])

        # visualize the goal
        self.sim.data.mocap_pos[0] = [x_target, 0, y_target]

        return self._get_obs()

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
            self.viewer.render()
        elif mode == 'rgb_array':
            if self.rgb_viewer is None:
                self.rgb_viewer = MjRenderContextOffscreen(self.sim, 0)
            self.rgb_viewer.render(500, 500)
            # window size used for old mujoco-py:
            data = self.rgb_viewer.read_pixels(500, 500, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
