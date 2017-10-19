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

import matplotlib.pyplot as plt
import numpy as np


def show_observation_stack(stack, channels_last=False):
    if isinstance(stack, list):  # is list
        stack_size = len(stack)
    elif len(stack.shape) == 3:
        stack_size = stack.shape[0]  # is numpy array
    elif len(stack.shape) == 4:
        stack_size = stack.shape[1]  # ignore batch dimension
        stack = stack[0]
    else:
        assert False, ""

    if channels_last:
        stack = np.transpose(stack, (2, 0, 1))
        stack_size = stack.shape[0]

    for i in range(stack_size):
        plt.subplot(1, stack_size, i + 1)
        plt.imshow(stack[i], cmap='gray')

    plt.show()


def show_diff_between_two_observations(observation1, observation2):
    plt.imshow(observation1 - observation2, cmap='gray')
    plt.show()


def plot_grayscale_observation(observation):
    plt.imshow(observation, cmap='gray')
    plt.show()
