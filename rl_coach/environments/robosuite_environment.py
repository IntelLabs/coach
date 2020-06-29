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

from typing import Union
from enum import Enum, Flag, auto
from copy import deepcopy
import numpy as np
import random
from collections import namedtuple

try:
    import robosuite
    from robosuite.wrappers import Wrapper
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("Robosuite")

from rl_coach.base_parameters import Parameters, VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import BoxActionSpace, VectorObservationSpace, StateSpace, PlanarMapsObservationSpace


class RobosuiteRobotType(Enum):
    SAWYER = 'Sawyer'
    PANDA = 'Panda'
    BAXTER = 'Baxter'
    IIWA = 'IIWA'
    JACO = 'Jaco'
    KINOVA3 = 'Kinova3'
    UR5e = 'UR5e'


_two_arm_robots = (RobosuiteRobotType.BAXTER,)
_one_arm_robots = tuple([r for r in RobosuiteRobotType if r not in _two_arm_robots])


class RobosuiteControllerType(Enum):
    JOINT_VELOCITY = "JOINT_VELOCITY"
    JOINT_TORQUE = "JOINT_TORQUE"
    JOINT_POSITION = "JOINT_POSITION"
    OSC_POSITION = "OSC_POSITION"
    OSC_POSE = "OSC_POSE"
    IK_POSE = "IK_POSE"


class OptionalObservations(Flag):
    NONE = 0
    CAMERA = auto()
    OBJECT = auto()


class RobosuiteCameraTypes(Enum):
    FRONT = 'frontview'
    BIRD = 'birdview'
    AGENT = 'agentview'


class RobosuiteBaseParameters(Parameters):
    def __init__(self, optional_observations: OptionalObservations = OptionalObservations.NONE):
        super(RobosuiteBaseParameters, self).__init__()
        # NOTE: Attribute names exactly match the attribute names in Robosuite
        self.horizon = 1000                 # Every episode lasts for exactly horizon timesteps
        self.ignore_done = True             # True if never terminating the environment (ignore horizon)
        self.reward_shaping = True          # if True, use dense rewards.
        self.gripper_visualizations = True  # True if using gripper visualization. Useful for teleoperation.
        self.use_indicator_object = False   # if True, sets up an indicator object that is useful for debugging

        # How many control signals to receive in every simulated second. This sets the amount of simulation time
        # that passes between every action input (this is NOT the same as frame_skip)
        self.control_freq = 10

        # Optional observations (robot state is always returned)
        # if True, every observation includes a rendered image
        self.use_camera_obs = bool(optional_observations & OptionalObservations.CAMERA)
        # if True, include object (cube/etc.) information in the observation
        self.use_object_obs = bool(optional_observations & OptionalObservations.OBJECT)

        # Camera parameters
        self.has_renderer = False
        self.has_offscreen_renderer = self.use_camera_obs
        self.render_collision_mesh = False              # True if rendering collision meshes in camera. False otherwise
        self.render_visual_mesh = True                  # True if rendering visual meshes in camera. False otherwise
        self.camera_names = RobosuiteCameraTypes.AGENT  # name of camera to be rendered (required for camera obs)
        self.camera_heights = 84                        # height of camera frame.
        self.camera_widths = 84                         # width of camera frame.
        self.camera_depths = False                      # True if rendering RGB-D, and RGB otherwise.

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
        return res


class RobosuiteTaskParameters(Parameters):
    def __init__(self):
        super(RobosuiteTaskParameters, self).__init__()

    def env_kwargs_dict(self):
        res = {k: (v.value if isinstance(v, Enum) else v) for k, v in vars(self).items()}
        return res


class RobosuiteLiftParameters(RobosuiteTaskParameters):
    def __init__(self):
        super(RobosuiteLiftParameters, self).__init__()
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (1., 5e-3, 1e-4)
        # self.placement_initializer = None


class RobosuiteSingleObjectMode(Enum):
    MULTI_OBJECT = 0
    SINGLE_RANDOM_PER_RESET = 1
    SINGLE_CONSTANT = 2


class RobosuitePickPlaceParameters(RobosuiteTaskParameters):
    class OBJECT_TYPES(Enum):
        MILK = 'milk'
        BREAD = 'bread'
        CEREAL = 'cereal'
        CAN = 'can'

    def __init__(self):
        super(RobosuitePickPlaceParameters, self).__init__()
        self.table_full_size = (0.39, 0.49, 0.82)
        self.table_friction = (1, 0.005, 0.0001)
        # self.placement_initializer = None
        self.single_object_mode = RobosuiteSingleObjectMode.MULTI_OBJECT
        self.object_type = None


class RobosuiteNutAssemblyParameters(RobosuiteTaskParameters):
    class NUT_TYPES(Enum):
        ROUND = 'round'
        SQUARE = 'square'

    def __init__(self):
        super(RobosuiteNutAssemblyParameters, self).__init__()
        self.table_full_size = (0.45, 0.69, 0.82)
        self.table_friction = (1, 0.005, 0.0001)
        # self.placement_initializer = None
        self.single_object_mode = RobosuiteSingleObjectMode.MULTI_OBJECT
        self.nut_type = None


class RobosuiteStackParameters(RobosuiteTaskParameters):
    def __init__(self):
        super(RobosuiteStackParameters, self).__init__()
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (1., 5e-3, 1e-4)
        # self.placement_initializer = None


class RobosuiteTwoArmLiftParameters(RobosuiteTaskParameters):
    def __init__(self):
        super(RobosuiteTwoArmLiftParameters, self).__init__()
        self.rescale_actions = True


class RobosuiteTwoArmPegInHoleParameters(RobosuiteTaskParameters):
    def __init__(self):
        super(RobosuiteTwoArmPegInHoleParameters, self).__init__()
        self.rescale_actions = True
        self.cylinder_radius = (0.015, 0.03)
        self.cylinder_length = 0.13


class RobosuiteLevels(Enum):
    LIFT = 'Lift'
    PICK_PLACE = 'PickPlace'
    NUT_ASSEMBLY = 'NutAssembly'
    STACK = 'Stack'
    TWO_ARM_LIFT = 'TwoArmLift'
    TWO_ARM_PEG_IN_HOLE = 'TwoArmPegInHole'


robosuite_level_expected_types = {
    RobosuiteLevels.LIFT: (RobosuiteLiftParameters, _one_arm_robots),
    RobosuiteLevels.PICK_PLACE: (RobosuitePickPlaceParameters, _one_arm_robots),
    RobosuiteLevels.NUT_ASSEMBLY: (RobosuiteNutAssemblyParameters, _one_arm_robots),
    RobosuiteLevels.STACK: (RobosuiteStackParameters, _one_arm_robots),
    RobosuiteLevels.TWO_ARM_LIFT: (RobosuiteTwoArmLiftParameters, _two_arm_robots),
    RobosuiteLevels.TWO_ARM_PEG_IN_HOLE: (RobosuiteTwoArmPegInHoleParameters, _two_arm_robots)
}


class RobosuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self, level, task_parameters, robot=None, controller=None):
        super().__init__(level=level)
        self.base_parameters = RobosuiteBaseParameters()
        self.task_parameters = task_parameters
        self.robot = robot
        self.controller = controller

    @property
    def path(self):
        return 'rl_coach.environments.robosuite_environment:RobosuiteEnvironment'


robosuite_envs = {env_name.lower(): env for env_name, env in robosuite.environments.REGISTERED_ENVS.items()}


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
                 seed: int, frame_skip: int, human_control: bool, custom_reward_threshold: Union[int, float],
                 visualization_parameters: VisualizationParameters,
                 base_parameters: RobosuiteBaseParameters,
                 task_parameters: RobosuiteTaskParameters,
                 robot: RobosuiteRobotType,
                 controller: RobosuiteControllerType,
                 target_success_rate: float = 1.0, **kwargs):
        super(RobosuiteEnvironment, self).__init__(level, seed, frame_skip, human_control, custom_reward_threshold,
                                                   visualization_parameters, target_success_rate)

        # Validate arguments

        self.frame_skip = max(1, self.frame_skip)
        base_parameters.horizon *= self.frame_skip

        try:
            self.level = RobosuiteLevels[self.env_id.upper()]
        except KeyError:
            raise ValueError("Unknown Robosuite level passed: '{}' ; Supported levels are: {}".format(
                level, ' | '.join([lvl.name.lower() for lvl in RobosuiteLevels])
            ))

        expected_params, expected_robots = robosuite_level_expected_types.get(self.level)
        if not isinstance(task_parameters, expected_params):
            raise TypeError("task_parameters for level '{}' must be of type {}".format(self.env_id,
                                                                                       expected_params.__name__))
        self.task_parameters = task_parameters

        if robot not in expected_robots:
            raise ValueError('{} robot not incompatible with level {} ; Compatible robots: {}'.format(
                robot, self.env_id, ' | '.join([str(r) for r in expected_robots])
            ))
        self.robot = robot

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
        env_args.update(task_parameters.env_kwargs_dict())
        env_args['robots'] = self.robot.value
        controller_cfg = None
        if controller is not None:
            controller_cfg = robosuite.controllers.load_controller_config(default_controller=self.controller.value)
        env_args['controller_configs'] = controller_cfg

        self.env: robosuite.environments.MujocoEnv = robosuite.make(self.level.value, **env_args)

        # TODO: Add DR
        # Wrap with a dummy wrapper so we get a consistent API (there are subtle changes between
        # wrappers and actual environments in Robosuite, for example action_spec as property vs. function)
        self.env = Wrapper(self.env)

        if isinstance(self.base_parameters.camera_names, Enum):
            self.base_parameters.camera_names = self.base_parameters.camera_names.value

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
        img: np.ndarray = self.env.sim.render(camera_name=RobosuiteCameraTypes.FRONT.value,
                                              height=512, width=512, depth=False)
        return np.flip(img, 0)

    def close(self):
        self.env.close()

