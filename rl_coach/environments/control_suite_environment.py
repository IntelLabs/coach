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
from enum import Enum
from typing import Union

import numpy as np

try:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("DeepMind Control Suite")

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.spaces import BoxActionSpace, ImageObservationSpace, VectorObservationSpace, StateSpace


class ObservationType(Enum):
    Measurements = 1
    Image = 2
    Image_and_Measurements = 3


# Parameters
class ControlSuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self, level=None):
        super().__init__(level=level)
        self.observation_type = ObservationType.Measurements
        self.default_input_filter = ControlSuiteInputFilter
        self.default_output_filter = ControlSuiteOutputFilter

    @property
    def path(self):
        return 'rl_coach.environments.control_suite_environment:ControlSuiteEnvironment'


"""
ControlSuite Environment Components
"""
ControlSuiteInputFilter = NoInputFilter()
ControlSuiteOutputFilter = NoOutputFilter()

control_suite_envs = {':'.join(env): ':'.join(env) for env in suite.BENCHMARKING}


# Environment
class ControlSuiteEnvironment(Environment):
    def __init__(self, level: LevelSelection, frame_skip: int, visualization_parameters: VisualizationParameters,
                 seed: Union[None, int]=None, human_control: bool=False,
                 observation_type: ObservationType=ObservationType.Measurements,
                 custom_reward_threshold: Union[int, float]=None, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        self.observation_type = observation_type

        # load and initialize environment
        domain_name, task_name = self.env_id.split(":")
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})

        if observation_type != ObservationType.Measurements:
            self.env = pixels.Wrapper(self.env, pixels_only=observation_type == ObservationType.Image)

        # seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.state_space = StateSpace({})

        # image observations
        if observation_type != ObservationType.Measurements:
            self.state_space['pixels'] = ImageObservationSpace(shape=self.env.observation_spec()['pixels'].shape,
                                                               high=255)

        # measurements observations
        if observation_type != ObservationType.Image:
            measurements_space_size = 0
            measurements_names = []
            for observation_space_name, observation_space in self.env.observation_spec().items():
                if len(observation_space.shape) == 0:
                    measurements_space_size += 1
                    measurements_names.append(observation_space_name)
                elif len(observation_space.shape) == 1:
                    measurements_space_size += observation_space.shape[0]
                    measurements_names.extend(["{}_{}".format(observation_space_name, i) for i in
                                               range(observation_space.shape[0])])
            self.state_space['measurements'] = VectorObservationSpace(shape=measurements_space_size,
                                                                      measurements_names=measurements_names)

        # actions
        self.action_space = BoxActionSpace(
            shape=self.env.action_spec().shape[0],
            low=self.env.action_spec().minimum,
            high=self.env.action_spec().maximum
        )

        # initialize the state by getting a new state from the environment
        self.reset_internal_state(True)

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            scale = 1
            if self.human_control:
                scale = 2
            if not self.native_rendering:
                self.renderer.create_screen(image.shape[1]*scale, image.shape[0]*scale)

    def _update_state(self):
        self.state = {}

        if self.observation_type != ObservationType.Measurements:
            self.pixels = self.last_result.observation['pixels']
            self.state['pixels'] = self.pixels

        if self.observation_type != ObservationType.Image:
            self.measurements = np.array([])
            for sub_observation in self.last_result.observation.values():
                if isinstance(sub_observation, np.ndarray) and len(sub_observation.shape) == 1:
                    self.measurements = np.concatenate((self.measurements, sub_observation))
                else:
                    self.measurements = np.concatenate((self.measurements, np.array([sub_observation])))
            self.state['measurements'] = self.measurements

        self.reward = self.last_result.reward if self.last_result.reward is not None else 0

        self.done = self.last_result.last()

    def _take_action(self, action):
        if type(self.action_space) == BoxActionSpace:
            action = self.action_space.clip_action_to_space(action)

        self.last_result = self.env.step(action)

    def _restart_environment_episode(self, force_environment_reset=False):
        self.last_result = self.env.reset()

    def _render(self):
        pass

    def get_rendered_image(self):
        return self.env.physics.render(camera_id=0)
