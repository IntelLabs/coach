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
from typing import Union, List

import numpy as np

from rl_coach.filters.observation.observation_move_axis_filter import ObservationMoveAxisFilter

try:
    from pysc2 import maps
    from pysc2.env import sc2_env
    from pysc2.env import available_actions_printer
    from pysc2.lib import actions
    from pysc2.lib import features
    from pysc2.env import environment
    from absl import app
    from absl import flags
except ImportError:
    from rl_coach.logger import failed_imports
    failed_imports.append("PySc2")

from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.spaces import BoxActionSpace, VectorObservationSpace, PlanarMapsObservationSpace, StateSpace, CompoundActionSpace, \
    DiscreteActionSpace
from rl_coach.filters.filter import InputFilter, OutputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.action.linear_box_to_box_map import LinearBoxToBoxMap
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter

FLAGS = flags.FLAGS
FLAGS(['coach.py'])

SCREEN_SIZE = 84  # will also impact the action space size

# Starcraft Constants
_NOOP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class StarcraftObservationType(Enum):
    Features = 0
    RGB = 1


StarcraftInputFilter = InputFilter(is_a_reference_filter=True)
StarcraftInputFilter.add_observation_filter('screen', 'move_axis', ObservationMoveAxisFilter(0, -1))
StarcraftInputFilter.add_observation_filter('screen', 'rescaling',
                                            ObservationRescaleToSizeFilter(
                                                PlanarMapsObservationSpace(np.array([84, 84, 1]),
                                                                           low=0, high=255, channels_axis=-1)))
StarcraftInputFilter.add_observation_filter('screen', 'to_uint8', ObservationToUInt8Filter(0, 255))

StarcraftInputFilter.add_observation_filter('minimap', 'move_axis', ObservationMoveAxisFilter(0, -1))
StarcraftInputFilter.add_observation_filter('minimap', 'rescaling',
                                            ObservationRescaleToSizeFilter(
                                                PlanarMapsObservationSpace(np.array([64, 64, 1]),
                                                                           low=0, high=255, channels_axis=-1)))
StarcraftInputFilter.add_observation_filter('minimap', 'to_uint8', ObservationToUInt8Filter(0, 255))


StarcraftNormalizingOutputFilter = OutputFilter(is_a_reference_filter=True)
StarcraftNormalizingOutputFilter.add_action_filter(
    'normalization', LinearBoxToBoxMap(input_space_low=-SCREEN_SIZE / 2, input_space_high=SCREEN_SIZE / 2 - 1))


class StarCraft2EnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.screen_size = 84
        self.minimap_size = 64
        self.feature_minimap_maps_to_use = range(7)
        self.feature_screen_maps_to_use = range(17)
        self.observation_type = StarcraftObservationType.Features
        self.disable_fog = False
        self.auto_select_all_army = True
        self.default_input_filter = StarcraftInputFilter
        self.default_output_filter = StarcraftNormalizingOutputFilter
        self.use_full_action_space = False


    @property
    def path(self):
        return 'rl_coach.environments.starcraft2_environment:StarCraft2Environment'


# Environment
class StarCraft2Environment(Environment):
    def __init__(self, level: LevelSelection, frame_skip: int, visualization_parameters: VisualizationParameters,
                 seed: Union[None, int]=None, human_control: bool=False,
                 custom_reward_threshold: Union[int, float]=None,
                 screen_size: int=84, minimap_size: int=64,
                 feature_minimap_maps_to_use: List=range(7), feature_screen_maps_to_use: List=range(17),
                 observation_type: StarcraftObservationType=StarcraftObservationType.Features,
                 disable_fog: bool=False, auto_select_all_army: bool=True,
                 use_full_action_space: bool=False, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.feature_minimap_maps_to_use = feature_minimap_maps_to_use
        self.feature_screen_maps_to_use = feature_screen_maps_to_use
        self.observation_type = observation_type
        self.features_screen_size = None
        self.feature_minimap_size = None
        self.rgb_screen_size = None
        self.rgb_minimap_size = None
        if self.observation_type == StarcraftObservationType.Features:
            self.features_screen_size = screen_size
            self.feature_minimap_size = minimap_size
        elif self.observation_type == StarcraftObservationType.RGB:
            self.rgb_screen_size = screen_size
            self.rgb_minimap_size = minimap_size
        self.disable_fog = disable_fog
        self.auto_select_all_army = auto_select_all_army
        self.use_full_action_space = use_full_action_space

        # step_mul is the equivalent to frame skipping. Not sure if it repeats actions in between or not though.
        self.env = sc2_env.SC2Env(map_name=self.env_id, step_mul=frame_skip,
                                  visualize=self.is_rendered,
                                  agent_interface_format=sc2_env.AgentInterfaceFormat(
                                      feature_dimensions=sc2_env.Dimensions(
                                          screen=self.features_screen_size,
                                          minimap=self.feature_minimap_size
                                      )
                                      # rgb_dimensions=sc2_env.Dimensions(
                                      #     screen=self.rgb_screen_size,
                                      #     minimap=self.rgb_screen_size
                                      # )
                                  ),
                                  # feature_screen_size=self.features_screen_size,
                                  # feature_minimap_size=self.feature_minimap_size,
                                  # rgb_screen_size=self.rgb_screen_size,
                                  # rgb_minimap_size=self.rgb_screen_size,
                                  disable_fog=disable_fog,
                                  random_seed=self.seed
                                  )

        # print all the available actions
        # self.env = available_actions_printer.AvailableActionsPrinter(self.env)

        self.reset_internal_state(True)

        """
        feature_screen:  [height_map, visibility_map, creep, power, player_id, player_relative, unit_type, selected,
                          unit_hit_points, unit_hit_points_ratio, unit_energy, unit_energy_ratio, unit_shields, 
                          unit_shields_ratio, unit_density, unit_density_aa, effects]
        feature_minimap: [height_map, visibility_map, creep, camera, player_id, player_relative, selecte
        d]
        player:          [player_id, minerals, vespene, food_cap, food_army, food_workers, idle_worker_dount, 
                          army_count, warp_gate_count, larva_count]
        """
        self.screen_shape = np.array(self.env.observation_spec()[0]['feature_screen'])
        self.screen_shape[0] = len(self.feature_screen_maps_to_use)
        self.minimap_shape = np.array(self.env.observation_spec()[0]['feature_minimap'])
        self.minimap_shape[0] = len(self.feature_minimap_maps_to_use)
        self.state_space = StateSpace({
            "screen": PlanarMapsObservationSpace(shape=self.screen_shape, low=0, high=255, channels_axis=0),
            "minimap": PlanarMapsObservationSpace(shape=self.minimap_shape, low=0, high=255, channels_axis=0),
            "measurements": VectorObservationSpace(self.env.observation_spec()[0]["player"][0])
        })
        if self.use_full_action_space:
            action_identifiers = list(self.env.action_spec()[0].functions)
            num_action_identifiers = len(action_identifiers)
            action_arguments = [(arg.name, arg.sizes) for arg in self.env.action_spec()[0].types]
            sub_action_spaces = [DiscreteActionSpace(num_action_identifiers)]
            for argument in action_arguments:
                for dimension in argument[1]:
                    sub_action_spaces.append(DiscreteActionSpace(dimension))
            self.action_space = CompoundActionSpace(sub_action_spaces)
        else:
            self.action_space = BoxActionSpace(2, 0, self.screen_size - 1, ["X-Axis, Y-Axis"],
                                               default_action=np.array([self.screen_size/2, self.screen_size/2]))

    def _update_state(self):
        timestep = 0
        self.screen = self.last_result[timestep].observation.feature_screen
        # extract only the requested segmentation maps from the observation
        self.screen = np.take(self.screen, self.feature_screen_maps_to_use, axis=0)
        self.minimap = self.last_result[timestep].observation.feature_minimap
        self.measurements = self.last_result[timestep].observation.player
        self.reward = self.last_result[timestep].reward
        self.done = self.last_result[timestep].step_type == environment.StepType.LAST
        self.state = {
            'screen': self.screen,
            'minimap': self.minimap,
            'measurements': self.measurements
        }

    def _take_action(self, action):
        if self.use_full_action_space:
            action_identifier = action[0]
            action_arguments = action[1:]
            action = actions.FunctionCall(action_identifier, action_arguments)
        else:
            coord = np.array(action[0:2])
            noop = False
            coord = coord.round()
            coord = np.clip(coord, 0, SCREEN_SIZE - 1)
            self.last_action_idx = coord

            if noop:
                action = actions.FunctionCall(_NOOP, [])
            else:
                action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])

        self.last_result = self.env.step(actions=[action])

    def _restart_environment_episode(self, force_environment_reset=False):
        # reset the environment
        self.last_result = self.env.reset()

        # select all the units on the screen
        if self.auto_select_all_army:
            self.env.step(actions=[actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

    def get_rendered_image(self):
        screen = np.squeeze(np.tile(np.expand_dims(self.screen, -1), (1, 1, 3)))
        screen = screen / np.max(screen) * 255
        return screen.astype('uint8')

    def dump_video_of_last_episode(self):
        from rl_coach.logger import experiment_path
        self.env._run_config.replay_dir = experiment_path
        self.env.save_replay('replays')
        super().dump_video_of_last_episode()
