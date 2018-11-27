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

from rl_coach.core_types import RunPhase, ActionType
from rl_coach.exploration_policies.exploration_policy import ExplorationPolicy, ExplorationParameters
from rl_coach.schedules import Schedule, LinearSchedule
from rl_coach.spaces import ActionSpace, BoxActionSpace


# TODO: consider renaming to gaussian sampling
class AdditiveNoiseParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.noise_percentage_schedule = LinearSchedule(0.1, 0.1, 50000)
        self.evaluation_noise_percentage = 0.05

    @property
    def path(self):
        return 'rl_coach.exploration_policies.additive_noise:AdditiveNoise'


class AdditiveNoise(ExplorationPolicy):
    """
    AdditiveNoise is an exploration policy intended for continuous action spaces. It takes the action from the agent
    and adds a Gaussian distributed noise to it. The amount of noise added to the action follows the noise amount that
    can be given in two different ways:
    1. Specified by the user as a noise schedule which is taken in percentiles out of the action space size
    2. Specified by the agents action. In case the agents action is a list with 2 values, the 1st one is assumed to
    be the mean of the action, and 2nd is assumed to be its standard deviation.
    """
    def __init__(self, action_space: ActionSpace, noise_percentage_schedule: Schedule,
                 evaluation_noise_percentage: float):
        """
        :param action_space: the action space used by the environment
        :param noise_percentage_schedule: the schedule for the noise variance percentage relative to the absolute range
                                          of the action space
        :param evaluation_noise_percentage: the noise variance percentage that will be used during evaluation phases
        """
        super().__init__(action_space)
        self.noise_percentage_schedule = noise_percentage_schedule
        self.evaluation_noise_percentage = evaluation_noise_percentage

        if not isinstance(action_space, BoxActionSpace):
            raise ValueError("Additive noise exploration works only for continuous controls."
                             "The given action space is of type: {}".format(action_space.__class__.__name__))

        if not np.all(-np.inf < action_space.high) or not np.all(action_space.high < np.inf)\
                or not np.all(-np.inf < action_space.low) or not np.all(action_space.low < np.inf):
            raise ValueError("Additive noise exploration requires bounded actions")

        # TODO: allow working with unbounded actions by defining the noise in terms of range and not percentage

    def get_action(self, action_values: List[ActionType]) -> ActionType:
        # TODO-potential-bug consider separating internally defined stdev and externally defined stdev into 2 policies

        # set the current noise percentage
        if self.phase == RunPhase.TEST:
            current_noise_precentage = self.evaluation_noise_percentage
        else:
            current_noise_precentage = self.noise_percentage_schedule.current_value

        # scale the noise to the action space range
        action_values_std = current_noise_precentage * (self.action_space.high - self.action_space.low)

        # extract the mean values
        if isinstance(action_values, list):
            # the action values are expected to be a list with the action mean and optionally the action stdev
            action_values_mean = action_values[0].squeeze()
        else:
            # the action values are expected to be a numpy array representing the action mean
            action_values_mean = action_values.squeeze()

        # step the noise schedule
        if self.phase == RunPhase.TRAIN:
            self.noise_percentage_schedule.step()
            # the second element of the list is assumed to be the standard deviation
            if isinstance(action_values, list) and len(action_values) > 1:
                action_values_std = action_values[1].squeeze()

        # add noise to the action means
        action = np.random.normal(action_values_mean, action_values_std)

        return action

    def get_control_param(self):
        return np.ones(self.action_space.shape)*self.noise_percentage_schedule.current_value
