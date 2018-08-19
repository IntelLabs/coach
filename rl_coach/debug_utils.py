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

import math

import matplotlib.pyplot as plt
import numpy as np

from rl_coach.filters.observation.observation_stacking_filter import LazyStack


def show_observation_stack(stack, channels_last=True, show=True, force_num_rows=None, row_to_update=0):
    if isinstance(stack, LazyStack):
        stack = np.array(stack)
    if isinstance(stack, list):  # is list
        stack_size = len(stack)
    elif len(stack.shape) == 3:
        stack_size = stack.shape[0]  # is numpy array
    elif len(stack.shape) == 4:
        stack_size = stack.shape[1]  # ignore batch dimension
        stack = stack[0]
    else:
        raise ValueError("The observation stack must be a list, a numpy array or a LazyStack object")

    if channels_last:
        stack = np.transpose(stack, (2, 0, 1))
        stack_size = stack.shape[0]

    max_cols = 10
    if force_num_rows:
        rows = force_num_rows
    else:
        rows = math.ceil(stack_size / max_cols)
    cols = max_cols if stack_size > max_cols else stack_size

    for i in range(stack_size):
        plt.subplot(rows, cols, row_to_update * cols + i + 1)
        plt.imshow(stack[i], cmap='gray')

    if show:
        plt.show()


def show_diff_between_two_observations(observation1, observation2):
    plt.imshow(observation1 - observation2, cmap='gray')
    plt.show()


def plot_grayscale_observation(observation):
    plt.imshow(observation, cmap='gray')
    plt.show()


def plot_episode_states(episode_transitions, state_variable: str='state', observation_index_in_stack: int=0):
    observations = []
    for transition in episode_transitions:
        observations.append(np.array(getattr(transition, state_variable)['observation'])[..., observation_index_in_stack])
    show_observation_stack(observations, False)


def plot_list_of_observation_stacks(observation_stacks):
    for idx, stack in enumerate(observation_stacks):
        show_observation_stack(stack['observation'], True, False,
                               force_num_rows=len(observation_stacks), row_to_update=idx)
    plt.show()
