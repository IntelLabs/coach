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

from enum import Enum

import gym
import numpy as np
from gym import spaces


class ExplorationChain(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30
    }

    class ObservationType(Enum):
        OneHot = 0
        Therm = 1

    def __init__(self, chain_length=16, start_state=1, max_steps=None, observation_type=ObservationType.Therm,
                 left_state_reward=1/1000, right_state_reward=1, simple_render=True):
        super().__init__()
        if chain_length <= 3:
            raise ValueError('Chain length must be > 3, found {}'.format(chain_length))
        if not 0 <= start_state < chain_length:
            raise ValueError('The start state should be within the chain bounds, found {}'.format(start_state))
        self.chain_length = chain_length
        self.start_state = start_state
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.left_state_reward = left_state_reward
        self.right_state_reward = right_state_reward
        self.simple_render = simple_render

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Discrete(2)  # 0 -> Go left, 1 -> Go right
        self.observation_space = spaces.Box(0, 1, shape=(chain_length,))#spaces.MultiBinary(chain_length)

        self.reset()

    def _terminate(self):
        return self.steps >= self.max_steps

    def _reward(self):
        if self.state == 0:
            return self.left_state_reward
        elif self.state == self.chain_length - 1:
            return self.right_state_reward
        else:
            return 0

    def step(self, action):
        # action is 0 or 1
        if action == 0:
            if 0 < self.state:
                self.state -= 1
        elif action == 1:
            if self.state < self.chain_length - 1:
                self.state += 1
        else:
            raise ValueError("An invalid action was given. The available actions are - 0 or 1, found {}".format(action))

        self.steps += 1

        return self._get_obs(), self._reward(), self._terminate(), {}

    def reset(self):
        self.steps = 0

        self.state = self.start_state

        return self._get_obs()

    def _get_obs(self):
        self.observation = np.zeros((self.chain_length,))
        if self.observation_type == self.ObservationType.OneHot:
            self.observation[self.state] = 1
        elif self.observation_type == self.ObservationType.Therm:
            self.observation[:(self.state+1)] = 1

        return self.observation

    def render(self, mode='human', close=False):
        if self.simple_render:
            observation = np.zeros((20, 20*self.chain_length))
            observation[:, self.state*20:(self.state+1)*20] = 255
            return observation
        else:
            # lazy loading of networkx and matplotlib to allow using the environment without installing them if
            # necessary
            import networkx as nx
            from networkx.drawing.nx_agraph import graphviz_layout
            import matplotlib.pyplot as plt

            if not hasattr(self, 'G'):
                self.states = list(range(self.chain_length))
                self.G = nx.DiGraph(directed=True)
                for i, origin_state in enumerate(self.states):
                    if i < self.chain_length - 1:
                        self.G.add_edge(origin_state,
                                        origin_state + 1,
                                        weight=0.5)
                    if i > 0:
                        self.G.add_edge(origin_state,
                                        origin_state - 1,
                                        weight=0.5, )
                    if i == 0 or i < self.chain_length - 1:
                        self.G.add_edge(origin_state,
                                        origin_state,
                                        weight=0.5, )

            fig = plt.gcf()
            if np.all(fig.get_size_inches() != [10, 2]):
                fig.set_size_inches(5, 1)
            color = ['y']*(len(self.G))
            color[self.state] = 'r'
            options = {
                'node_color': color,
                'node_size': 50,
                'width': 1,
                'arrowstyle': '-|>',
                'arrowsize': 5,
                'font_size': 6
            }
            pos = graphviz_layout(self.G, prog='dot', args='-Grankdir=LR')
            nx.draw_networkx(self.G, pos, arrows=True, **options)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data
