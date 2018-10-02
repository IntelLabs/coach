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
from collections import OrderedDict
from typing import Union

import pygame
from pandas import to_pickle

from rl_coach.agents.agent import Agent
from rl_coach.agents.bc_agent import BCNetworkParameters
from rl_coach.architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, EmbedderScheme, \
    AgentParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.core_types import ActionInfo
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.logger import screen
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters


class HumanAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()


class HumanNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.input_embedders_parameters['observation'].scheme = EmbedderScheme.Medium
        self.middleware_parameters = FCMiddlewareParameters()
        self.optimizer_type = 'Adam'
        self.batch_size = 32
        self.replace_mse_with_huber_loss = False
        self.create_target_network = False


class HumanAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=HumanAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={"main": BCNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.human_agent:HumanAgent'


class HumanAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

        self.clock = pygame.time.Clock()
        self.max_fps = int(self.ap.visualization.max_fps_for_human_control)
        self.env = None

    def init_environment_dependent_modules(self):
        super().init_environment_dependent_modules()
        self.env = self.parent_level_manager._real_environment
        screen.log_title("Human Control Mode")
        available_keys = self.env.get_available_keys()
        if available_keys:
            screen.log("Use keyboard keys to move. Press escape to quit. Available keys:")
            screen.log("")
            for action, key in self.env.get_available_keys():
                screen.log("\t- {}: {}".format(action, key))
            screen.separator()

    def train(self):
        return 0

    def choose_action(self, curr_state):
        action = ActionInfo(self.env.get_action_from_user(), action_value=0)
        action = self.output_filter.reverse_filter(action)

        # keep constant fps
        self.clock.tick(self.max_fps)

        if not self.env.renderer.is_open:
            self.save_replay_buffer_and_exit()

        return action

    def save_replay_buffer_and_exit(self):
        replay_buffer_path = os.path.join(self.agent_logger.experiments_path, 'replay_buffer.p')
        self.memory.tp = None
        self.memory.save(replay_buffer_path)
        screen.log_title("Replay buffer was stored in {}".format(replay_buffer_path))
        exit()

    def log_to_screen(self):
        # log to screen
        log = OrderedDict()
        log["Episode"] = self.current_episode
        log["Total reward"] = round(self.total_reward_in_current_episode, 2)
        log["Steps"] = self.total_steps_counter
        screen.log_dict(log, prefix="Recording")
