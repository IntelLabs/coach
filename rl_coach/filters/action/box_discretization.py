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

from itertools import product
from typing import Union, List

import numpy as np

from rl_coach.filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace


class BoxDiscretization(PartialDiscreteActionSpaceMap):
    """
    Discretizes a continuous action space into a discrete action space, allowing the usage of
    agents such as DQN for continuous environments such as MuJoCo. Given the number of bins to discretize into, the
    original continuous action space is uniformly separated into the given number of bins, each mapped to a discrete
    action index. Each discrete action is mapped to a single N dimensional action in the BoxActionSpace action space.
    For example, if the original actions space is between -1 and 1 and 5 bins were selected, the new action
    space will consist of 5 actions mapped to -1, -0.5, 0, 0.5 and 1.
    """
    def __init__(self, num_bins_per_dimension: Union[int, List[int]], force_int_bins=False):
        """
        :param num_bins_per_dimension: The number of bins to use for each dimension of the target action space.
                                       The bins will be spread out uniformly over this space
        :param force_int_bins: force the bins to represent only integer actions. for example, if the action space is in
                               the range 0-10 and there are 5 bins, then the bins will be placed at 0, 2, 5, 7, 10,
                               instead of 0, 2.5, 5, 7.5, 10.
        """
        # we allow specifying either a single number for all dimensions, or a single number per dimension in the target
        # action space
        self.num_bins_per_dimension = num_bins_per_dimension
        self.force_int_bins = force_int_bins
        super().__init__()

    def validate_output_action_space(self, output_action_space: BoxActionSpace):
        if not isinstance(output_action_space, BoxActionSpace):
            raise ValueError("BoxActionSpace discretization only works with an output space of type BoxActionSpace. "
                             "The given output space is {}".format(output_action_space))

        if len(self.num_bins_per_dimension) != output_action_space.shape:
            # TODO: this check is not sufficient. it does not deal with actions spaces with more than one axis
            raise ValueError("The length of the list of bins per dimension ({}) does not match the number of "
                             "dimensions in the action space ({})"
                             .format(len(self.num_bins_per_dimension), output_action_space))

    def get_unfiltered_action_space(self, output_action_space: BoxActionSpace) -> DiscreteActionSpace:
        if isinstance(self.num_bins_per_dimension, int):
            self.num_bins_per_dimension = np.ones(output_action_space.shape) * self.num_bins_per_dimension

        bins = []
        for i in range(len(output_action_space.low)):
            dim_bins = np.linspace(output_action_space.low[i], output_action_space.high[i],
                                   self.num_bins_per_dimension[i])
            if self.force_int_bins:
                dim_bins = dim_bins.astype(int)
            bins.append(dim_bins)
        self.target_actions = [list(action) for action in list(product(*bins))]

        return super().get_unfiltered_action_space(output_action_space)
