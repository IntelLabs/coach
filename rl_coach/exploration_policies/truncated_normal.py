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

from typing import List

import numpy as np
from scipy.stats import truncnorm

from rl_coach.core_types import RunPhase, ActionType
from rl_coach.exploration_policies.exploration_policy import ExplorationParameters, ContinuousActionExplorationPolicy
from rl_coach.schedules import Schedule, LinearSchedule
from rl_coach.spaces import ActionSpace, BoxActionSpace


class TruncatedNormalParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.noise_schedule = LinearSchedule(0.1, 0.1, 50000)
        self.evaluation_noise = 0.05
        self.clip_low = 0
        self.clip_high = 1
        self.noise_as_percentage_from_action_space = True

    @property
    def path(self):
        return 'rl_coach.exploration_policies.truncated_normal:TruncatedNormal'


class TruncatedNormal(ContinuousActionExplorationPolicy):
    """
    The TruncatedNormal exploration policy is intended for continuous action spaces. It samples the action from a
    normal distribution, where the mean action is given by the agent, and the standard deviation can be given in t
    wo different ways:
    1. Specified by the user as a noise schedule which is taken in percentiles out of the action space size
    2. Specified by the agents action. In case the agents action is a list with 2 values, the 1st one is assumed to
    be the mean of the action, and 2nd is assumed to be its standard deviation.
    When the sampled action is outside of the action bounds given by the user, it is sampled again and again, until it
    is within the bounds.
    """
    def __init__(self, action_space: ActionSpace, noise_schedule: Schedule,
                 evaluation_noise: float, clip_low: float, clip_high: float,
                 noise_as_percentage_from_action_space: bool = True):
        """
        :param action_space: the action space used by the environment
        :param noise_schedule: the schedule for the noise variance
        :param evaluation_noise: the noise variance that will be used during evaluation phases
        :param noise_as_percentage_from_action_space: whether to consider the noise as a percentage of the action space
                                                        or absolute value
        """
        super().__init__(action_space)
        self.noise_schedule = noise_schedule
        self.evaluation_noise = evaluation_noise
        self.noise_as_percentage_from_action_space = noise_as_percentage_from_action_space
        self.clip_low = clip_low
        self.clip_high = clip_high

        if not isinstance(action_space, BoxActionSpace):
            raise ValueError("Truncated normal exploration works only for continuous controls."
                             "The given action space is of type: {}".format(action_space.__class__.__name__))

        if not np.all(-np.inf < action_space.high) or not np.all(action_space.high < np.inf)\
                or not np.all(-np.inf < action_space.low) or not np.all(action_space.low < np.inf):
            raise ValueError("Additive noise exploration requires bounded actions")

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        # set the current noise
        if self.phase == RunPhase.TEST:
            current_noise = self.evaluation_noise
        else:
            current_noise = self.noise_schedule.current_value

        # scale the noise to the action space range
        if self.noise_as_percentage_from_action_space:
            action_values_std = current_noise * (self.action_space.high - self.action_space.low)
        else:
            action_values_std = current_noise

        # extract the mean values
        if isinstance(action_values, list):
            # the action values are expected to be a list with the action mean and optionally the action stdev
            action_values_mean = action_values[0].squeeze()
        else:
            # the action values are expected to be a numpy array representing the action mean
            action_values_mean = action_values.squeeze()

        # step the noise schedule
        if self.phase is not RunPhase.TEST:
            self.noise_schedule.step()
            # the second element of the list is assumed to be the standard deviation
            if isinstance(action_values, list) and len(action_values) > 1:
                action_values_std = action_values[1].squeeze()

        # sample from truncated normal distribution
        normalized_low = (self.clip_low - action_values_mean) / action_values_std
        normalized_high = (self.clip_high - action_values_mean) / action_values_std
        distribution = truncnorm(normalized_low, normalized_high, loc=action_values_mean, scale=action_values_std)
        action = distribution.rvs(1)

        return action

    def get_control_param(self):
        return np.ones(self.action_space.shape)*self.noise_schedule.current_value
