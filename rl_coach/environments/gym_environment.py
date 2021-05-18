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

import gym
import numpy as np
from enum import IntEnum
import scipy.ndimage

from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.utils import lower_under_to_upper, short_dynamic_import

try:
    import roboschool
    from OpenGL import GL
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("RoboSchool")

try:
    from gym_extensions.continuous import mujoco
except:
    from rl_coach.logger import failed_imports
    failed_imports.append("GymExtensions")

try:
    import pybullet_envs
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("PyBullet")

from typing import Dict, Any, Union
from rl_coach.core_types import RunPhase, EnvironmentSteps
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace, ImageObservationSpace, VectorObservationSpace, \
    PlanarMapsObservationSpace, TensorObservationSpace, StateSpace, RewardSpace
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.filters.filter import InputFilter
import random
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.logger import screen


# Parameters
class GymEnvironmentParameters(EnvironmentParameters):
    def __init__(self, level=None):
        super().__init__(level=level)
        self.random_initialization_steps = 0
        self.max_over_num_frames = 1
        self.additional_simulator_parameters = {}
        self.observation_space_type = None

    @property
    def path(self):
        return 'rl_coach.environments.gym_environment:GymEnvironment'


# Generic parameters for vector environments such as mujoco, roboschool, bullet, etc.
class GymVectorEnvironment(GymEnvironmentParameters):
    def __init__(self, level=None):
        super().__init__(level=level)
        self.frame_skip = 1
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()


# Roboschool
gym_roboschool_envs = ['inverted_pendulum', 'inverted_pendulum_swingup', 'inverted_double_pendulum', 'reacher',
                       'hopper', 'walker2d', 'half_cheetah', 'ant', 'humanoid', 'humanoid_flagrun',
                       'humanoid_flagrun_harder', 'pong']
roboschool_v1 = {e: "Roboschool{}".format(lower_under_to_upper(e) + '-v1') for e in gym_roboschool_envs}

# Mujoco
gym_mujoco_envs = ['inverted_pendulum', 'inverted_double_pendulum', 'reacher', 'hopper', 'walker2d', 'half_cheetah',
                   'ant', 'swimmer', 'humanoid', 'humanoid_standup', 'pusher', 'thrower', 'striker']

mujoco_v2 = {e: "{}".format(lower_under_to_upper(e) + '-v2') for e in gym_mujoco_envs}
mujoco_v2['walker2d'] = 'Walker2d-v2'

# Fetch
gym_fetch_envs = ['reach', 'slide', 'push', 'pick_and_place']
fetch_v1 = {e: "{}".format('Fetch' + lower_under_to_upper(e) + '-v1') for e in gym_fetch_envs}


"""
Atari Environment Components
"""

AtariInputFilter = InputFilter(is_a_reference_filter=True)
AtariInputFilter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
AtariInputFilter.add_observation_filter('observation', 'rescaling',
                                        ObservationRescaleToSizeFilter(ImageObservationSpace(np.array([84, 84, 3]),
                                                                                             high=255)))
AtariInputFilter.add_observation_filter('observation', 'to_grayscale', ObservationRGBToYFilter())
AtariInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
AtariInputFilter.add_observation_filter('observation', 'stacking', ObservationStackingFilter(4))
AtariOutputFilter = NoOutputFilter()


class Atari(GymEnvironmentParameters):
    def __init__(self, level=None):
        super().__init__(level=level)
        self.frame_skip = 4
        self.max_over_num_frames = 2
        self.random_initialization_steps = 30
        self.default_input_filter = AtariInputFilter
        self.default_output_filter = AtariOutputFilter


gym_atari_envs = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
                  'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                  'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
                  'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
                  'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
                  'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
                  'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
                  'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
                  'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']
atari_deterministic_v4 = {e: "{}".format(lower_under_to_upper(e) + 'Deterministic-v4') for e in gym_atari_envs}
atari_no_frameskip_v4 = {e: "{}".format(lower_under_to_upper(e) + 'NoFrameskip-v4') for e in gym_atari_envs}


# default atari schedule used in the DeepMind papers
atari_schedule = ScheduleParameters()
atari_schedule.improve_steps = EnvironmentSteps(50000000)
atari_schedule.steps_between_evaluation_periods = EnvironmentSteps(250000)
atari_schedule.evaluation_steps = EnvironmentSteps(135000)
atari_schedule.heatup_steps = EnvironmentSteps(50000)


class MaxOverFramesAndFrameskipEnvWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=4, max_over_num_frames=2):
        super().__init__(env)
        self.max_over_num_frames = max_over_num_frames
        self.observations_stack = []
        self.frameskip = frameskip
        self.first_frame_to_max_over = self.frameskip - self.max_over_num_frames

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        self.observations_stack = []
        for i in range(self.frameskip):
            observation, reward, done, info = self.env.step(action)
            if i >= self.first_frame_to_max_over:
                self.observations_stack.append(observation)
            total_reward += reward
            if done:
                # deal with last state in episode
                if not self.observations_stack:
                    self.observations_stack.append(observation)
                break

        max_over_frames_observation = np.max(self.observations_stack, axis=0)

        return max_over_frames_observation, total_reward, done, info


# Environment
class ObservationSpaceType(IntEnum):
    Tensor = 0
    Image = 1
    Vector = 2


class GymEnvironment(Environment):
    def __init__(self,
                 level: LevelSelection,
                 frame_skip: int,
                 visualization_parameters: VisualizationParameters,
                 target_success_rate: float=1.0,
                 additional_simulator_parameters: Dict[str, Any] = {},
                 seed: Union[None, int] = None,
                 human_control: bool=False,
                 custom_reward_threshold: Union[int, float]=None,
                 random_initialization_steps: int=1,
                 max_over_num_frames: int=1,
                 observation_space_type: ObservationSpaceType=None,
                 **kwargs):
        """
        :param level: (str)
            A string representing the gym level to run. This can also be a LevelSelection object.
            For example, BreakoutDeterministic-v0

        :param frame_skip: (int)
            The number of frames to skip between any two actions given by the agent. The action will be repeated
            for all the skipped frames.

        :param visualization_parameters: (VisualizationParameters)
            The parameters used for visualizing the environment, such as the render flag, storing videos etc.

        :param additional_simulator_parameters: (Dict[str, Any])
            Any additional parameters that the user can pass to the Gym environment. These parameters should be
            accepted by the __init__ function of the implemented Gym environment.

        :param seed: (int)
            A seed to use for the random number generator when running the environment.

        :param human_control: (bool)
            A flag that allows controlling the environment using the keyboard keys.

        :param custom_reward_threshold: (float)
            Allows defining a custom reward that will be used to decide when the agent succeeded in passing the environment.
            If not set, this value will be taken from the Gym environment definition.

        :param random_initialization_steps: (int)
            The number of random steps that will be taken in the environment after each reset.
            This is a feature presented in the DQN paper, which improves the variability of the episodes the agent sees.

        :param max_over_num_frames: (int)
            This value will be used for merging multiple frames into a single frame by taking the maximum value for each
            of the pixels in the frame. This is particularly used in Atari games, where the frames flicker, and objects
            can be seen in one frame but disappear in the next.

        :param observation_space_type:
            This value will be used for generating observation space. Allows a custom space. Should be one of
            ObservationSpaceType. If not specified, observation space is inferred from the number of dimensions
            of the observation: 1D: Vector space, 3D: Image space if 1 or 3 channels, PlanarMaps space otherwise.
        """
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold,
                         visualization_parameters, target_success_rate)

        self.random_initialization_steps = random_initialization_steps
        self.max_over_num_frames = max_over_num_frames
        self.additional_simulator_parameters = additional_simulator_parameters

        # hide warnings
        gym.logger.set_level(40)

        """
        load and initialize environment
        environment ids can be defined in 3 ways:
        1. Native gym environments like BreakoutDeterministic-v0 for example
        2. Custom gym environments written and installed as python packages.
           This environments should have a python module with a class inheriting gym.Env, implementing the
           relevant functions (_reset, _step, _render) and defining the observation and action space
           For example: my_environment_package:MyEnvironmentClass will run an environment defined in the
           MyEnvironmentClass class
        3. Custom gym environments written as an independent module which is not installed.
           This environments should have a python module with a class inheriting gym.Env, implementing the
           relevant functions (_reset, _step, _render) and defining the observation and action space.
           For example: path_to_my_environment.sub_directory.my_module:MyEnvironmentClass will run an
           environment defined in the MyEnvironmentClass class which is located in the module in the relative path
           path_to_my_environment.sub_directory.my_module
        """
        if ':' in self.env_id:
            # custom environments
            if '/' in self.env_id or '.' in self.env_id:
                # environment in a an absolute path module written as a unix path or in a relative path module
                # written as a python import path
                env_class = short_dynamic_import(self.env_id)
            else:
                # environment in a python package
                env_class = gym.envs.registration.load(self.env_id)

            # instantiate the environment
            try:
                self.env = env_class(**self.additional_simulator_parameters)
            except:
                screen.error("Failed to instantiate Gym environment class %s with arguments %s" %
                             (env_class, self.additional_simulator_parameters), crash=False)
                raise
        else:
            self.env = gym.make(self.env_id)

        # for classic control we want to use the native renderer because otherwise we will get 2 renderer windows
        environment_to_always_use_with_native_rendering = ['classic_control', 'mujoco', 'robotics']
        self.native_rendering = self.native_rendering or \
                                any([env in str(self.env.unwrapped.__class__)
                                     for env in environment_to_always_use_with_native_rendering])
        if self.native_rendering:
            if hasattr(self, 'renderer'):
                self.renderer.close()

        # seed
        if self.seed is not None:
            self.env.seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # frame skip and max between consecutive frames
        self.is_mujoco_env = 'mujoco' in str(self.env.unwrapped.__class__)
        self.is_roboschool_env = 'roboschool' in str(self.env.unwrapped.__class__)
        self.is_atari_env = 'Atari' in str(self.env.unwrapped.__class__)
        if self.is_atari_env:
            self.env.unwrapped.frameskip = 1  # this accesses the atari env that is wrapped with a timelimit wrapper env
            if self.env_id == "SpaceInvadersDeterministic-v4" and self.frame_skip == 4:
                screen.warning("Warning: The frame-skip for Space Invaders was automatically updated from 4 to 3. "
                               "This is following the DQN paper where it was noticed that a frame-skip of 3 makes the "
                               "laser rays disappear. To force frame-skip of 4, please use SpaceInvadersNoFrameskip-v4.")
                self.frame_skip = 3
            self.env = MaxOverFramesAndFrameskipEnvWrapper(self.env,
                                                           frameskip=self.frame_skip,
                                                           max_over_num_frames=self.max_over_num_frames)
        else:
            self.env.unwrapped.frameskip = self.frame_skip

        self.state_space = StateSpace({})

        # observations
        if not isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            state_space = {'observation': self.env.observation_space}
        else:
            state_space = self.env.observation_space.spaces

        for observation_space_name, observation_space in state_space.items():
            if observation_space_type == ObservationSpaceType.Tensor:
                # we consider arbitrary input tensor which does not necessarily represent images
                self.state_space[observation_space_name] = TensorObservationSpace(
                    shape=np.array(observation_space.shape),
                    low=observation_space.low,
                    high=observation_space.high
                )
            elif observation_space_type == ObservationSpaceType.Image or len(observation_space.shape) == 3:
                # we assume gym has image observations (with arbitrary number of channels) where their values are
                # within 0-255, and where the channel dimension is the last dimension
                if observation_space.shape[-1] in [1, 3]:
                    self.state_space[observation_space_name] = ImageObservationSpace(
                        shape=np.array(observation_space.shape),
                        high=255,
                        channels_axis=-1
                    )
                else:
                    # For any number of channels other than 1 or 3, use the generic PlanarMaps space
                    self.state_space[observation_space_name] = PlanarMapsObservationSpace(
                        shape=np.array(observation_space.shape),
                        low=0,
                        high=255,
                        channels_axis=-1
                    )
            elif observation_space_type == ObservationSpaceType.Vector or len(observation_space.shape) == 1:
                self.state_space[observation_space_name] = VectorObservationSpace(
                    shape=observation_space.shape[0],
                    low=observation_space.low,
                    high=observation_space.high
                )
            else:
                raise screen.error("Failed to instantiate Gym environment class %s with observation space type %s" %
                                 (env_class, observation_space_type), crash=True)

        if 'desired_goal' in state_space.keys():
            self.goal_space = self.state_space['desired_goal']

        # actions
        if type(self.env.action_space) == gym.spaces.box.Box:
            self.action_space = BoxActionSpace(
                shape=self.env.action_space.shape,
                low=self.env.action_space.low,
                high=self.env.action_space.high
            )
        elif type(self.env.action_space) == gym.spaces.discrete.Discrete:
            actions_description = []
            if hasattr(self.env.unwrapped, 'get_action_meanings'):
                actions_description = self.env.unwrapped.get_action_meanings()
            self.action_space = DiscreteActionSpace(
                num_actions=self.env.action_space.n,
                descriptions=actions_description
            )
        else:
            raise screen.error((
                "Failed to instantiate gym environment class {} due to unsupported "
                "action space {}. Expected BoxActionSpace or DiscreteActionSpace."
            ).format(env_class, self.env.action_space), crash=True)

        if self.human_control:
            # TODO: add this to the action space
            # map keyboard keys to actions
            self.key_to_action = {}
            if hasattr(self.env.unwrapped, 'get_keys_to_action'):
                self.key_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                screen.error("Error: Environment {} does not support human control.".format(self.env), crash=True)

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            scale = 1
            if self.human_control:
                scale = 2
            if not self.native_rendering:
                self.renderer.create_screen(image.shape[1]*scale, image.shape[0]*scale)

        # the info is only updated after the first step
        self.state_space['measurements'] = VectorObservationSpace(shape=len(self.info.keys()))

        if self.env.spec and custom_reward_threshold is None:
                self.reward_success_threshold = self.env.spec.reward_threshold
                self.reward_space = RewardSpace(1, reward_success_threshold=self.reward_success_threshold)

        self.target_success_rate = target_success_rate

    def _wrap_state(self, state):
        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            return {'observation': state}
        return state

    def _update_state(self):
        if self.is_atari_env and hasattr(self, 'current_ale_lives') \
                and self.current_ale_lives != self.env.unwrapped.ale.lives():
            if self.phase == RunPhase.TRAIN or self.phase == RunPhase.HEATUP:
                # signal termination for life loss
                self.done = True
            elif self.phase == RunPhase.TEST and not self.done:
                # the episode is not terminated in evaluation, but we need to press fire again
                self._press_fire()
            self._update_ale_lives()
        # TODO: update the measurements
        if self.state and "desired_goal" in self.state.keys():
            self.goal = self.state['desired_goal']

    def _take_action(self, action):
        if type(self.action_space) == BoxActionSpace:
            action = self.action_space.clip_action_to_space(action)

        self.state, self.reward, self.done, self.info = self.env.step(action)
        self.state = self._wrap_state(self.state)

    def _random_noop(self):
        # simulate a random initial environment state by stepping for a random number of times between 0 and 30
        step_count = 0
        random_initialization_steps = random.randint(0, self.random_initialization_steps)
        while self.action_space is not None and (self.state is None or step_count < random_initialization_steps):
            step_count += 1
            self.step(self.action_space.default_action)

    def _press_fire(self):
        fire_action = 1
        if self.is_atari_env and self.env.unwrapped.get_action_meanings()[fire_action] == 'FIRE':
            self.current_ale_lives = self.env.unwrapped.ale.lives()
            self.step(fire_action)
            if self.done:
                self.reset_internal_state()

    def _update_ale_lives(self):
        if self.is_atari_env:
            self.current_ale_lives = self.env.unwrapped.ale.lives()

    def _restart_environment_episode(self, force_environment_reset=False):
        # prevent reset of environment if there are ale lives left
        if (self.is_atari_env and self.env.unwrapped.ale.lives() > 0) \
                and not force_environment_reset and self.env.env._elapsed_steps < self.env.env._max_episode_steps:
            self.step(self.action_space.default_action)
        else:
            self.state = self.env.reset()
            self.state = self._wrap_state(self.state)
            self._update_ale_lives()

        if self.is_atari_env:
            self._random_noop()
            self._press_fire()

        # initialize the number of lives
        self._update_ale_lives()

    def _render(self):
        self.env.render(mode='human')

    def get_rendered_image(self):
        image = self.env.render(mode='rgb_array')
        return image

    def get_target_success_rate(self) -> float:
        return self.target_success_rate

    def close(self) -> None:
        """
        Clean up to close rendering windows.

        :return: None
        """
        self.env.close()
