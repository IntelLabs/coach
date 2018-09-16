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

import operator
import time
from collections import OrderedDict
from typing import Union, List, Tuple, Dict

import numpy as np

from rl_coach import logger
from rl_coach.base_parameters import Parameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import GoalType, ActionType, EnvResponse, RunPhase
from rl_coach.environments.environment_interface import EnvironmentInterface
from rl_coach.logger import screen
from rl_coach.renderer import Renderer
from rl_coach.spaces import ActionSpace, ObservationSpace, DiscreteActionSpace, RewardSpace, StateSpace
from rl_coach.utils import squeeze_list, force_list


class LevelSelection(object):
    def __init__(self, level: str):
        self.selected_level = level

    def select(self, level: str):
        self.selected_level = level

    def __str__(self):
        if self.selected_level is None:
            logger.screen.error("No level has been selected. Please select a level using the -lvl command line flag, "
                                "or change the level in the preset.", crash=True)
        return self.selected_level


class SingleLevelSelection(LevelSelection):
    def __init__(self, levels: Union[str, List[str], Dict[str, str]]):
        super().__init__(None)
        self.levels = levels
        if isinstance(levels, list):
            self.levels = {level: level for level in levels}
        if isinstance(levels, str):
            self.levels = {levels: levels}

    def __str__(self):
        if self.selected_level is None:
            logger.screen.error("No level has been selected. Please select a level using the -lvl command line flag, "
                                "or change the level in the preset. \nThe available levels are: \n{}"
                                .format(', '.join(sorted(self.levels.keys()))), crash=True)
        if self.selected_level not in self.levels.keys():
            logger.screen.error("The selected level ({}) is not part of the available levels ({})"
                                .format(self.selected_level, ', '.join(self.levels.keys())), crash=True)
        return self.levels[self.selected_level]


# class SingleLevelPerPhase(LevelSelection):
#     def __init__(self, levels: Dict[RunPhase, str]):
#         super().__init__(None)
#         self.levels = levels
#
#     def __str__(self):
#         super().__str__()
#         if self.selected_level not in self.levels.keys():
#             logger.screen.error("The selected level ({}) is not part of the available levels ({})"
#                                 .format(self.selected_level, self.levels.keys()), crash=True)
#         return self.levels[self.selected_level]


class CustomWrapper(object):
    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return getattr(self.environment, attr, False)


class EnvironmentParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.level = None
        self.frame_skip = 4
        self.seed = None
        self.human_control = False
        self.custom_reward_threshold = None
        self.default_input_filter = None
        self.default_output_filter = None
        self.experiment_path = None

    @property
    def path(self):
        return 'rl_coach.environments.environment:Environment'


class Environment(EnvironmentInterface):
    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        """
        :param level: The environment level. Each environment can have multiple levels
        :param seed: a seed for the random number generator of the environment
        :param frame_skip: number of frames to skip (while repeating the same action) between each two agent directives
        :param human_control: human should control the environment
        :param visualization_parameters: a blob of parameters used for visualization of the environment
        :param **kwargs: as the class is instantiated by EnvironmentParameters, this is used to support having
                         additional arguments which will be ignored by this class, but might be used by others
        """
        super().__init__()

        # env initialization

        self.game = []

        self.state = {}
        self.observation = None
        self.goal = None
        self.reward = 0
        self.done = False
        self.info = {}
        self._last_env_response = None
        self.last_action = 0
        self.episode_idx = 0
        self.total_steps_counter = 0
        self.current_episode_steps_counter = 0
        self.last_episode_time = time.time()
        self.key_to_action = {}
        self.last_episode_images = []

        # rewards
        self.total_reward_in_current_episode = 0
        self.max_reward_achieved = -np.inf
        self.reward_success_threshold = custom_reward_threshold

        # spaces
        self.state_space = self._state_space = None
        self.goal_space = self._goal_space = None
        self.action_space = self._action_space = None
        self.reward_space = RewardSpace(1, reward_success_threshold=self.reward_success_threshold)  # TODO: add a getter and setter

        self.env_id = str(level)
        self.seed = seed
        self.frame_skip = frame_skip

        # human interaction and visualization
        self.human_control = human_control
        self.wait_for_explicit_human_action = False
        self.is_rendered = visualization_parameters.render or self.human_control
        self.native_rendering = visualization_parameters.native_rendering and not self.human_control
        self.visualization_parameters = visualization_parameters
        if not self.native_rendering:
            self.renderer = Renderer()

    @property
    def action_space(self) -> Union[List[ActionSpace], ActionSpace]:
        """
        Get the action space of the environment
        :return: the action space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, val: Union[List[ActionSpace], ActionSpace]):
        """
        Set the action space of the environment
        :return: None
        """
        self._action_space = val

    @property
    def state_space(self) -> Union[List[StateSpace], StateSpace]:
        """
        Get the state space of the environment
        :return: the observation space
        """
        return self._state_space

    @state_space.setter
    def state_space(self, val: Union[List[StateSpace], StateSpace]):
        """
        Set the state space of the environment
        :return: None
        """
        self._state_space = val

    @property
    def goal_space(self) -> Union[List[ObservationSpace], ObservationSpace]:
        """
        Get the state space of the environment
        :return: the observation space
        """
        return self._goal_space

    @goal_space.setter
    def goal_space(self, val: Union[List[ObservationSpace], ObservationSpace]):
        """
        Set the goal space of the environment
        :return: None
        """
        self._goal_space = val

    def get_action_from_user(self) -> ActionType:
        """
        Get an action from the user keyboard
        :return: action index
        """
        if self.wait_for_explicit_human_action:
            while len(self.renderer.pressed_keys) == 0:
                self.renderer.get_events()

        if self.key_to_action == {}:
            # the keys are the numbers on the keyboard corresponding to the action index
            if len(self.renderer.pressed_keys) > 0:
                action_idx = self.renderer.pressed_keys[0] - ord("1")
                if 0 <= action_idx < self.action_space.shape[0]:
                    return action_idx
        else:
            # the keys are mapped through the environment to more intuitive keyboard keys
            # key = tuple(self.renderer.pressed_keys)
            # for key in self.renderer.pressed_keys:
            for env_keys in self.key_to_action.keys():
                if set(env_keys) == set(self.renderer.pressed_keys):
                    return self.action_space.actions[self.key_to_action[env_keys]]

        # return the default action 0 so that the environment will continue running
        return self.action_space.default_action

    @property
    def last_env_response(self) -> Union[List[EnvResponse], EnvResponse]:
        """
        Get the last environment response
        :return: a dictionary that contains the state, reward, etc.
        """
        return squeeze_list(self._last_env_response)

    @last_env_response.setter
    def last_env_response(self, val: Union[List[EnvResponse], EnvResponse]):
        """
        Set the last environment response
        :param val: the last environment response
        """
        self._last_env_response = force_list(val)

    def step(self, action: ActionType) -> EnvResponse:
        """
        Make a single step in the environment using the given action
        :param action: an action to use for stepping the environment. Should follow the definition of the action space.
        :return: the environment response as returned in get_last_env_response
        """
        action = self.action_space.clip_action_to_space(action)
        if self.action_space and not self.action_space.val_matches_space_definition(action):
            raise ValueError("The given action does not match the action space definition. "
                             "Action = {}, action space definition = {}".format(action, self.action_space))

        # store the last agent action done and allow passing None actions to repeat the previously done action
        if action is None:
            action = self.last_action
        self.last_action = action
        if self.visualization_parameters.add_rendered_image_to_env_response:
            current_rendered_image = self.get_rendered_image()

        self.current_episode_steps_counter += 1
        if self.phase != RunPhase.UNDEFINED:
            self.total_steps_counter += 1

        # act
        self._take_action(action)

        # observe
        self._update_state()

        if self.is_rendered:
            self.render()

        self.total_reward_in_current_episode += self.reward

        if self.visualization_parameters.add_rendered_image_to_env_response:
            self.info['image'] = current_rendered_image

        self.last_env_response = \
            EnvResponse(
                reward=self.reward,
                next_state=self.state,
                goal=self.goal,
                game_over=self.done,
                info=self.info
            )

        # store observations for video / gif dumping
        if self.should_dump_video_of_the_current_episode(episode_terminated=False) and \
            (self.visualization_parameters.dump_mp4 or self.visualization_parameters.dump_gifs):
            self.last_episode_images.append(self.get_rendered_image())

        return self.last_env_response

    def render(self) -> None:
        """
        Call the environment function for rendering to the screen
        """
        if self.native_rendering:
            self._render()
        else:
            self.renderer.render_image(self.get_rendered_image())

    def reset_internal_state(self, force_environment_reset=False) -> EnvResponse:
        """
        Reset the environment and all the variable of the wrapper
        :param force_environment_reset: forces environment reset even when the game did not end
        :return: A dictionary containing the observation, reward, done flag, action and measurements
        """

        self.dump_video_of_last_episode_if_needed()
        self._restart_environment_episode(force_environment_reset)
        self.last_episode_time = time.time()

        if self.current_episode_steps_counter > 0 and self.phase != RunPhase.UNDEFINED:
            self.episode_idx += 1

        self.done = False
        self.total_reward_in_current_episode = self.reward = 0.0
        self.last_action = 0
        self.current_episode_steps_counter = 0
        self.last_episode_images = []
        self._update_state()

        # render before the preprocessing of the observation, so that the image will be in its original quality
        if self.is_rendered:
            self.render()

        self.last_env_response = \
            EnvResponse(
                reward=self.reward,
                next_state=self.state,
                goal=self.goal,
                game_over=self.done,
                info=self.info
            )

        return self.last_env_response

    def get_random_action(self) -> ActionType:
        """
        Returns an action picked uniformly from the available actions
        :return: a numpy array with a random action
        """
        return self.action_space.sample()

    def get_available_keys(self) -> List[Tuple[str, ActionType]]:
        """
        Return a list of tuples mapping between action names and the keyboard key that triggers them
        :return: a list of tuples mapping between action names and the keyboard key that triggers them
        """
        available_keys = []
        if self.key_to_action != {}:
            for key, idx in sorted(self.key_to_action.items(), key=operator.itemgetter(1)):
                if key != ():
                    key_names = [self.renderer.get_key_names([k])[0] for k in key]
                    available_keys.append((self.action_space.descriptions[idx], ' + '.join(key_names)))
        elif type(self.action_space) == DiscreteActionSpace:
            for action in range(self.action_space.shape):
                available_keys.append(("Action {}".format(action + 1), action + 1))
        return available_keys

    def get_goal(self) -> GoalType:
        """
        Get the current goal that the agents needs to achieve in the environment
        :return: The goal
        """
        return self.goal

    def set_goal(self, goal: GoalType) -> None:
        """
        Set the current goal that the agent needs to achieve in the environment
        :param goal: the goal that needs to be achieved
        :return: None
        """
        self.goal = goal

    def should_dump_video_of_the_current_episode(self, episode_terminated=False):
        if self.visualization_parameters.video_dump_methods:
            for video_dump_method in force_list(self.visualization_parameters.video_dump_methods):
                if not video_dump_method.should_dump(episode_terminated, **self.__dict__):
                    return False
            return True
        return False

    def dump_video_of_last_episode_if_needed(self):
        if self.visualization_parameters.video_dump_methods and self.last_episode_images != []:
            if self.should_dump_video_of_the_current_episode(episode_terminated=True):
                self.dump_video_of_last_episode()

    def dump_video_of_last_episode(self):
        frame_skipping = max(1, int(5 / self.frame_skip))
        file_name = 'episode-{}_score-{}'.format(self.episode_idx, self.total_reward_in_current_episode)
        fps = 10
        if self.visualization_parameters.dump_gifs:
            logger.create_gif(self.last_episode_images[::frame_skipping], name=file_name, fps=fps)
        if self.visualization_parameters.dump_mp4:
            logger.create_mp4(self.last_episode_images[::frame_skipping], name=file_name, fps=fps)

    def log_to_screen(self):
        # log to screen
        log = OrderedDict()
        log["Episode"] = self.episode_idx
        log["Total reward"] = np.round(self.total_reward_in_current_episode, 2)
        log["Steps"] = self.total_steps_counter
        screen.log_dict(log, prefix=self.phase.value)

    # The following functions define the interaction with the environment.
    # Any new environment that inherits the Environment class should use these signatures.
    # Some of these functions are optional - please read their description for more details.

    def _take_action(self, action_idx: ActionType) -> None:
        """
        An environment dependent function that sends an action to the simulator.
        :param action_idx: the action to perform on the environment
        :return: None
        """
        raise NotImplementedError("")

    def _update_state(self) -> None:
        """
        Updates the state from the environment.
        Should update self.observation, self.reward, self.done, self.measurements and self.info
        :return: None
        """
        raise NotImplementedError("")

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        """
        Restarts the simulator episode
        :param force_environment_reset: Force the environment to reset even if the episode is not done yet.
        :return: None
        """
        raise NotImplementedError("")

    def _render(self) -> None:
        """
        Renders the environment using the native simulator renderer
        :return: None
        """
        pass

    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        return np.transpose(self.state['observation'], [1, 2, 0])


"""
Video Dumping Methods
"""


class VideoDumpMethod(object):
    """
    Method used to decide when to dump videos
    """
    def should_dump(self, episode_terminated=False, **kwargs):
        raise NotImplementedError("")


class AlwaysDumpMethod(VideoDumpMethod):
    """
    Dump video for every episode
    """
    def __init__(self):
        super().__init__()

    def should_dump(self, episode_terminated=False, **kwargs):
        return True


class MaxDumpMethod(VideoDumpMethod):
    """
    Dump video every time a new max total reward has been achieved
    """
    def __init__(self):
        super().__init__()
        self.max_reward_achieved = -np.inf

    def should_dump(self, episode_terminated=False, **kwargs):
        # if the episode has not finished yet we want to be prepared for dumping a video
        if not episode_terminated:
            return True
        if kwargs['total_reward_in_current_episode'] > self.max_reward_achieved:
            self.max_reward_achieved = kwargs['total_reward_in_current_episode']
            return True
        else:
            return False


class EveryNEpisodesDumpMethod(object):
    """
    Dump videos once in every N episodes
    """
    def __init__(self, num_episodes_between_dumps: int):
        super().__init__()
        self.num_episodes_between_dumps = num_episodes_between_dumps
        self.last_dumped_episode = 0
        if num_episodes_between_dumps < 1:
            raise ValueError("the number of episodes between dumps should be a positive number")

    def should_dump(self, episode_terminated=False, **kwargs):
        if kwargs['episode_idx'] >= self.last_dumped_episode + self.num_episodes_between_dumps - 1:
            self.last_dumped_episode = kwargs['episode_idx']
            return True
        else:
            return False


class SelectedPhaseOnlyDumpMethod(object):
    """
    Dump videos when the phase of the environment matches a predefined phase
    """
    def __init__(self, run_phases: Union[RunPhase, List[RunPhase]]):
        self.run_phases = force_list(run_phases)

    def should_dump(self, episode_terminated=False, **kwargs):
        if kwargs['_phase'] in self.run_phases:
            return True
        else:
            return False
