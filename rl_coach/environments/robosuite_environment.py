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

try:
    import robosuite
    from robosuite.wrappers import IKWrapper
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("Robosuite")

from rl_coach.base_parameters import Parameters, VisualizationParameters
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.spaces import BoxActionSpace, ImageObservationSpace, VectorObservationSpace, StateSpace


class RobosuiteRobotType(Enum):
    SAWYER = 'Sawyer'
    PANDA = 'Panda'
    BAXTER = 'Baxter'


_one_arm_robots = (RobosuiteRobotType.SAWYER, RobosuiteRobotType.PANDA)
_two_arm_robots = (RobosuiteRobotType.BAXTER,)


# class OptionalObservations(Flag):
#     NONE = 0
#     CAMERA = auto()
#     OBJECT = auto()


class RobosuiteCameraTypes(Enum):
    FRONT = 'frontview'
    BIRD = 'birdview'
    AGENT = 'agentview'


class RobosuiteBaseParameters(Parameters):
    def __init__(self):
        super(RobosuiteBaseParameters, self).__init__()
        # NOTE: Attribute names exactly match the attribute names in Robosuite
        self.horizon = 1000                 # Every episode lasts for exactly horizon timesteps
        self.ignore_done = False            # True if never terminating the environment (ignore horizon)
        self.reward_shaping = True          # if True, use dense rewards.
        self.gripper_visualization = True   # True if using gripper visualization. Useful for teleoperation.
        self.use_indicator_obj = False      # if True, sets up an indicator object that is useful for debugging

        # Optional observations (robot state is always returned)
        self.use_camera_obs = True          # if True, every observation includes a rendered image
        self.use_object_obs = True          # if True, include object (cube/etc.) information in the observation

        # Camera parameters
        self.render_collision_mesh = False              # True if rendering collision meshes in camera. False otherwise
        self.render_visual_mesh = True                  # True if rendering visual meshes in camera. False otherwise
        self.camera_name = RobosuiteCameraTypes.FRONT   # name of camera to be rendered (required for camera obs)
        self.camera_height = 256                        # height of camera frame.
        self.camera_width = 256                         # width of camera frame.
        self.camera_depth = False                       # True if rendering RGB-D, and RGB otherwise.

    def env_kwargs_dict(self):
        res = deepcopy(vars(self))
        res['camera_name'] = res['camera_name'].value
        return res


class RobosuiteTaskParameters(Parameters):
    def __init__(self):
        super(RobosuiteTaskParameters, self).__init__()

    def env_kwargs_dict(self):
        res = {k: (v.value if isinstance(v, Enum) else v) for k, v in vars(self)}
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
    LIFT = 'Lift',
    PICK_PLACE = 'PickPlace',
    NUT_ASSEMBLY = 'NutAssembly',
    STACK = 'Stack',
    TWO_ARM_LIFT = 'TwoArmLift',
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
    def __init__(self, level, task_parameters, robot):
        super().__init__(level=level)
        self.base_parameters = RobosuiteBaseParameters()
        self.task_parameters = task_parameters
        self.robot = robot

    @property
    def path(self):
        return 'rl_coach.environments.robosuite_environment:RobosuiteEnvironment'


robosuite_envs = {env_name.lower(): env for env_name, env in robosuite.environments.REGISTERED_ENVS.items()}


# Environment
class RobosuiteEnvironment(Environment):
    def __init__(self, level: LevelSelection,
                 seed: int, frame_skip: int, human_control: bool, custom_reward_threshold: Union[int, float],
                 visualization_parameters: VisualizationParameters,
                 base_parameters: RobosuiteBaseParameters,
                 task_parameters: RobosuiteTaskParameters,
                 robot: RobosuiteRobotType,
                 target_success_rate: float = 1.0, **kwargs):
        super(RobosuiteEnvironment, self).__init__(level, seed, frame_skip, human_control, custom_reward_threshold,
                                                   visualization_parameters, target_success_rate)

        # Validate arguments

        try:
            self.level = RobosuiteLevels[self.env_id.upper()]
        except KeyError:
            raise ValueError("Unknown Robosuite level passed: '{}' ; Supported levels are: {}".format(
                level, ', '.join([lvl.name.lower() for lvl in RobosuiteLevels])
            ))

        expected_params, expected_robots = robosuite_level_expected_types.get(self.level)
        if not isinstance(task_parameters, expected_params):
            raise TypeError("task_parameters for level '{}' must be of type ".format(self.env_id,
                                                                                     expected_params.__name__))
        self.task_parameters = task_parameters

        if robot not in expected_robots:
            raise ValueError('{} robot not incompatible with level {} ; Compatible robots: {}'.format(
                robot, self.env_id, ', '.join([r for r in RobosuiteRobotType])
            ))
        self.robot = robot

        # Load and initialize environment

        env_args = base_parameters.env_kwargs_dict()
        env_args.update(task_parameters.env_kwargs_dict())
        env_args['has_renderer'] = visualization_parameters.native_rendering

        robosuite_env_name = self.robot.value + self.level.replace('TwoArm', '')

        self.env: robosuite.environments.MujocoEnv = robosuite.make(robosuite_env_name, **env_args)

        # Robosuite doesn't have the concept of frame skip, instead it takes a "control frequency"
        # parameter which set the number of control (action) signals the simulator gets each
        # simulated second.
        mujoco_fps = 1. / self.env.model_timestep
        self.env.control_freq = mujoco_fps / frame_skip
        self.env.initialize_time(self.env.control_freq)

