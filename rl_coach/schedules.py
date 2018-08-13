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

from typing import List, Tuple

import numpy as np

from rl_coach.core_types import EnvironmentSteps


class Schedule(object):
    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        self.current_value = initial_value

    def step(self):
        raise NotImplementedError("")


class ConstantSchedule(Schedule):
    def __init__(self, initial_value: float):
        super().__init__(initial_value)

    def step(self):
        pass


class LinearSchedule(Schedule):
    """
    A simple linear schedule which decreases or increases over time from an initial to a final value
    """
    def __init__(self, initial_value: float, final_value: float, decay_steps: int):
        """
        :param initial_value: the initial value
        :param final_value: the final value
        :param decay_steps: the number of steps that are required to decay the initial value to the final value
        """
        super().__init__(initial_value)
        self.final_value = final_value
        self.decay_steps = decay_steps
        self.decay_delta = (initial_value - final_value) / float(decay_steps)

    def step(self):
        self.current_value -= self.decay_delta
        # decreasing schedule
        if self.final_value < self.initial_value:
            self.current_value = np.clip(self.current_value, self.final_value, self.initial_value)
        # increasing schedule
        if self.final_value > self.initial_value:
            self.current_value = np.clip(self.current_value, self.initial_value, self.final_value)


class PieceWiseSchedule(Schedule):
    """
    A schedule which consists of multiple sub-schedules, where each one is used for a defined number of steps
    """
    def __init__(self, schedules: List[Tuple[Schedule, EnvironmentSteps]]):
        """
        :param schedules: a list of schedules to apply serially. Each element of the list should be a tuple of
                          2 elements - a schedule and the number of steps to run it in terms of EnvironmentSteps
        """
        super().__init__(schedules[0][0].initial_value)
        self.schedules = schedules
        self.current_schedule = schedules[0]
        self.current_schedule_idx = 0
        self.current_schedule_step_count = 0

    def step(self):
        self.current_schedule[0].step()

        if self.current_schedule_idx < len(self.schedules) - 1 \
                and self.current_schedule_step_count >= self.current_schedule[1].num_steps:
            self.current_schedule_idx += 1
            self.current_schedule = self.schedules[self.current_schedule_idx]
            self.current_schedule_step_count = 0

        self.current_value = self.current_schedule[0].current_value
        self.current_schedule_step_count += 1


class ExponentialSchedule(Schedule):
    """
    A simple exponential schedule which decreases or increases over time from an initial to a final value
    """
    def __init__(self, initial_value: float, final_value: float, decay_coefficient: float):
        """
        :param initial_value: the initial value
        :param final_value: the final value
        :param decay_coefficient: the exponential decay coefficient
        """
        super().__init__(initial_value)
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_coefficient = decay_coefficient
        self.current_step = 0
        self.current_value = self.initial_value
        if decay_coefficient < 1 and final_value > initial_value:
            raise ValueError("The final value should be lower than the initial value when the decay coefficient < 1")
        if decay_coefficient > 1 and initial_value > final_value:
            raise ValueError("The final value should be higher than the initial value when the decay coefficient > 1")

    def step(self):
        self.current_value *= self.decay_coefficient

        # decreasing schedule
        if self.final_value < self.initial_value:
            self.current_value = np.clip(self.current_value, self.final_value, self.initial_value)
        # increasing schedule
        if self.final_value > self.initial_value:
            self.current_value = np.clip(self.current_value, self.initial_value, self.final_value)

        self.current_step += 1
