#
# Copyright (c) 2020 Intel Corporation
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

from typing import Union ,Dict, Any
from enum import Enum, Flag, auto
from copy import deepcopy
import numpy as np
import random
from collections import namedtuple

try:
    import robosuite
    from robosuite.wrappers import Wrapper, DomainRandomizationWrapper
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("Robosuite")

from rl_coach.base_parameters import Parameters, VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import BoxActionSpace, VectorObservationSpace, StateSpace, PlanarMapsObservationSpace


robosuite_environments = list(robosuite.ALL_ENVIRONMENTS)
robosuite_robots = list(robosuite.ALL_ROBOTS)
robosuite_controllers = list(robosuite.ALL_CONTROLLERS)


def get_robosuite_env_extra_parameters(env_name: str):
    import inspect
    assert env_name in robosuite_environments

    env_params = inspect.signature(robosuite.environments.REGISTERED_ENVS[env_name]).parameters
    base_params = list(RobosuiteBaseParameters().env_kwargs_dict().keys()) + ['robots', 'controller_configs']
    return {n: p.default for n, p in env_params.items() if n not in base_params}


class OptionalObservations(Flag):
    NONE = 0
    CAMERA = auto()
    OBJECT = auto()


class RobosuiteBaseParameters(Parameters):
    def __init__(self, optional_observations: OptionalObservations = OptionalObservations.NONE):
        super(RobosuiteBaseParameters, self).__init__()

        # NOTE: Attribute names should exactly match the attribute names in Robosuite

        self.horizon = 1000         # Every episode lasts for exactly horizon timesteps
        self.ignore_done = True     # True if never terminating the environment (ignore horizon)
        self.reward_scale = 0       # Scales the reward by the amount specified. Use 0 to take Robosuite default
        self.reward_shaping = True  # if True, use dense rewards.

        self.use_indicator_object = False  # if True, sets up an indicator object that is useful for debugging

        # How many control signals to receive in every simulated second. This sets the amount of simulation time
        # that passes between every action input (this is NOT the same as frame_skip)
        self.control_freq = 10

        # Optional observations (robot state is always returned)
        # if True, every observation includes a rendered image
        self.use_camera_obs = bool(optional_observations & OptionalObservations.CAMERA)
        # if True, include object (cube/etc.) information in the observation
        self.use_object_obs = bool(optional_observations & OptionalObservations.OBJECT)

        # removing joint velocities from the observation as it makes the sim2real transfer dynamics dependent
        self.use_joint_vel_obs = False

        # Camera parameters
        self.has_renderer = False            # Set to true to use Mujoco native viewer for on-screen rendering
        self.render_camera = 'frontview'     # name of camera to use for on-screen rendering
        self.has_offscreen_renderer = self.use_camera_obs
        self.gripper_visualizations = False  # True if using gripper visualization. Useful for teleoperation.
        self.render_collision_mesh = False   # True if rendering collision meshes in camera. False otherwise
        self.render_visual_mesh = True       # True if rendering visual meshes in camera. False otherwise
        self.camera_names = 'agentview'      # name of camera for rendering camera observations
        self.camera_heights = 84             # height of camera frame.
        self.camera_widths = 84              # width of camera frame.
        self.camera_depths = False           # True if rendering RGB-D, and RGB otherwise.

        # Collision
        self.penalize_on_collision = True
        self.end_episode_on_collision = False
        
    @property
    def optional_observations(self):
        flag = OptionalObservations.NONE
        if self.use_camera_obs:
            flag = OptionalObservations.CAMERA
            if self.use_object_obs:
                flag |= OptionalObservations.OBJECT
        elif self.use_object_obs:
            flag = OptionalObservations.OBJECT
        return flag

    @optional_observations.setter
    def optional_observations(self, value):
        self.use_camera_obs = bool(value & OptionalObservations.CAMERA)
        if self.use_camera_obs:
            self.has_offscreen_renderer = True
        self.use_object_obs = bool(value & OptionalObservations.OBJECT)

    def env_kwargs_dict(self):
        res = {k: (v.value if isinstance(v, Enum) else v) for k, v in vars(self).items()}
        if self.reward_scale == 0:
            res.pop('reward_scale')
        return res


class RobosuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self, level, robot=None, controller=None, apply_dr: bool = False,
                 dr_every_n_steps_min: int = 10, dr_every_n_steps_max: int = 20):
        super().__init__(level=level)
        self.base_parameters = RobosuiteBaseParameters()
        self.extra_parameters = {}
        self.robot = robot
        self.controller = controller
        self.apply_dr = apply_dr
        self.dr_every_n_steps_min = dr_every_n_steps_min
        self.dr_every_n_steps_max = dr_every_n_steps_max


    @property
    def path(self):
        return 'rl_coach.environments.robosuite_environment:RobosuiteEnvironment'


RobosuiteStepResult = namedtuple('RobosuiteStepResult', ['observation', 'reward', 'done', 'info'])


def _process_observation(raw_obs, camera_name):
    new_obs = {}

    camera_obs = raw_obs.get(camera_name + '_image', None)
    if camera_obs is not None:
        depth_obs = raw_obs.get(camera_name + '_depth', None)
        if depth_obs is not None:
            depth_obs = np.expand_dims(depth_obs, axis=2)
            camera_obs = np.concatenate([camera_obs, depth_obs], axis=2)
        new_obs['camera'] = camera_obs

    measurements = raw_obs['robot0_robot-state']
    object_obs = raw_obs.get('object-state', None)
    if object_obs is not None:
        measurements = np.concatenate([measurements, object_obs])
    new_obs['measurements'] = measurements

    return new_obs


# Environment
class RobosuiteEnvironment(Environment):
    def __init__(self, level: LevelSelection,
                 seed: int, frame_skip: int, human_control: bool, custom_reward_threshold: Union[int, float, None],
                 visualization_parameters: VisualizationParameters,
                 base_parameters: RobosuiteBaseParameters,
                 extra_parameters: Dict[str, Any],
                 robot: str,
                 controller: str,
                 target_success_rate: float = 1.0, task_id: int = 0, apply_dr: bool = False,
                 dr_every_n_steps_min: int = 10, dr_every_n_steps_max: int = 20, **kwargs):
        super(RobosuiteEnvironment, self).__init__(level, seed, frame_skip, human_control, custom_reward_threshold,
                                                   visualization_parameters, target_success_rate, task_id)

        # Validate arguments

        self.frame_skip = max(1, self.frame_skip)
        base_parameters.horizon *= self.frame_skip

        def validate_input(input, supported, name):
            if input not in supported:
                raise ValueError("Unknown Robosuite {0} passed: '{1}' ; Supported {0}s are: {2}".format(
                    name, input, ' | '.join(supported)
                ))

        validate_input(self.env_id, robosuite_environments, 'environment')
        validate_input(robot, robosuite_robots, 'robot')
        self.robot = robot
        if controller is not None:
            validate_input(controller, robosuite_controllers, 'controller')
        self.controller = controller

        self.base_parameters = base_parameters
        self.base_parameters.has_renderer = self.is_rendered and self.native_rendering
        self.base_parameters.has_offscreen_renderer = self.base_parameters.use_camera_obs or (self.is_rendered and not
                                                                                              self.native_rendering)

        # Seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Load and initialize environment
        env_args = self.base_parameters.env_kwargs_dict()
        env_args.update(extra_parameters)
        env_args['robots'] = self.robot
        controller_cfg = None
        if self.controller is not None:
            controller_cfg = robosuite.controllers.load_controller_config(default_controller=self.controller)
        env_args['controller_configs'] = controller_cfg

        self.env: robosuite.environments.MujocoEnv = robosuite.make(self.env_id, **env_args)

        # Wrap with a dummy wrapper so we get a consistent API (there are subtle changes between
        # wrappers and actual environments in Robosuite, for example action_spec as property vs. function)
        self.env = Wrapper(self.env)
        if apply_dr:
            self.env = DomainRandomizationWrapper(self.env, randomize_every_n_steps_min=dr_every_n_steps_min,
                                                  randomize_every_n_steps_max=dr_every_n_steps_max
                                                  )

        # State space
        self.state_space = StateSpace({})
        dummy_obs = _process_observation(self.env.observation_spec(), self.base_parameters.camera_names)

        self.state_space['measurements'] = VectorObservationSpace(dummy_obs['measurements'].shape[0])

        if self.base_parameters.use_camera_obs:
            self.state_space['camera'] = PlanarMapsObservationSpace(dummy_obs['camera'].shape, 0, 255)

        # Action space
        low, high = self.env.unwrapped.action_spec
        self.action_space = BoxActionSpace(low.shape, low=low, high=high)

        self.reset_internal_state()

        if self.is_rendered:
            image = self.get_rendered_image()
            self.renderer.create_screen(image.shape[1], image.shape[0])
        # TODO: Other environments call rendering here, why? reset_internal_state does it

    def _take_action(self, action):
        action = self.action_space.clip_action_to_space(action)

        # We mimic the "action_repeat" mechanism of RobosuiteWrapper in Surreal.
        # Same concept as frame_skip, only returning the average reward across repeated actions instead
        # of the total reward.
        rewards = []
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        reward = np.mean(rewards)
        self.last_result = RobosuiteStepResult(obs, reward, done, info)

    def _update_state(self):
        obs = _process_observation(self.last_result.observation, self.base_parameters.camera_names)
        self.state = {k: obs[k] for k in self.state_space.sub_spaces}
        self.reward = self.last_result.reward or 0
        self.done = self.last_result.done
        self.info = self.last_result.info

    def _restart_environment_episode(self, force_environment_reset=False):
        reset_obs = self.env.reset()
        self.last_result = RobosuiteStepResult(reset_obs, 0.0, False, {})

    def _render(self):
        self.env.render()

    def get_rendered_image(self):
        img: np.ndarray = self.env.sim.render(camera_name=self.base_parameters.render_camera,
                                              height=512, width=512, depth=False)
        return np.flip(img, 0)

    def close(self):
        self.env.close()

