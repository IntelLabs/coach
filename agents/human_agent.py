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
import collections
import os

import pygame
from pandas.io import pickle

from agents import agent
import logger
import utils


class HumanAgent(agent.Agent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        agent.Agent.__init__(self, env, tuning_parameters, replicated_device, thread_id)

        self.clock = pygame.time.Clock()
        self.max_fps = int(self.tp.visualization.max_fps_for_human_control)

        logger.screen.log_title("Human Control Mode")
        available_keys = self.env.get_available_keys()
        if available_keys:
            logger.screen.log("Use keyboard keys to move. Press escape to quit. Available keys:")
            logger.screen.log("")
            for action, key in self.env.get_available_keys():
                logger.screen.log("\t- {}: {}".format(action, key))
            logger.screen.separator()

    def train(self):
        return 0

    def choose_action(self, curr_state, phase=utils.RunPhase.TRAIN):
        action = self.env.get_action_from_user()

        # keep constant fps
        self.clock.tick(self.max_fps)

        if not self.env.renderer.is_open:
            self.save_replay_buffer_and_exit()

        return action, {"action_value": 0}

    def save_replay_buffer_and_exit(self):
        replay_buffer_path = os.path.join(logger.logger.experiments_path, 'replay_buffer.p')
        self.memory.tp = None
        pickle.to_pickle(self.memory, replay_buffer_path)
        logger.screen.log_title("Replay buffer was stored in {}".format(replay_buffer_path))
        exit()

    def log_to_screen(self, phase):
        # log to logger.screen
        logger.screen.log_dict(
            collections.OrderedDict([
                ("Episode", self.current_episode),
                ("total reward", self.total_reward_in_current_episode),
                ("steps", self.total_steps_counter)
            ]),
            prefix="Recording"
        )
