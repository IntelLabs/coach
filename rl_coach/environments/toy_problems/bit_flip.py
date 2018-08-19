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

import gym
import numpy as np
from gym import spaces


class BitFlip(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30
    }

    def __init__(self, bit_length=16, max_steps=None, mean_zero=False):
        super(BitFlip, self).__init__()
        if bit_length < 1:
            raise ValueError('bit_length must be >= 1, found {}'.format(bit_length))
        self.bit_length = bit_length
        self.mean_zero = mean_zero

        if max_steps is None:
            # default to bit_length
            self.max_steps = bit_length
        elif max_steps == 0:
            self.max_steps = None
        else:
            self.max_steps = max_steps

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Discrete(bit_length)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(bit_length, )),
            'desired_goal': spaces.Box(low=0, high=1, shape=(bit_length, )),
            'achieved_goal': spaces.Box(low=0, high=1, shape=(bit_length, ))
        })

        self.reset()

    def _terminate(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps

    def _reward(self):
        return -1 if (self.state != self.goal).any() else 0

    def step(self, action):
        # action is an int in the range [0, self.bit_length)
        self.state[action] = int(not self.state[action])
        self.steps += 1

        return (self._get_obs(), self._reward(), self._terminate(), {})

    def reset(self):
        self.steps = 0

        self.state = np.array([random.choice([1, 0]) for _ in range(self.bit_length)])

        # make sure goal is not the initial state
        self.goal = self.state
        while (self.goal == self.state).all():
            self.goal = np.array([random.choice([1, 0]) for _ in range(self.bit_length)])

        return self._get_obs()

    def _mean_zero(self, x):
        if self.mean_zero:
            return (x - 0.5) / 0.5
        else:
            return x

    def _get_obs(self):
        return {
            'state': self._mean_zero(self.state),
            'desired_goal': self._mean_zero(self.goal),
            'achieved_goal': self._mean_zero(self.state)
        }

    def render(self, mode='human', close=False):
        observation = np.zeros((20, 20 * self.bit_length, 3))
        for bit_idx, (state_bit, goal_bit) in enumerate(zip(self.state, self.goal)):
            # green if the bit matches
            observation[:, bit_idx * 20:(bit_idx + 1) * 20, 1] = (state_bit == goal_bit) * 255
            # red if the bit doesn't match
            observation[:, bit_idx * 20:(bit_idx + 1) * 20, 0] = (state_bit != goal_bit) * 255
        return observation
