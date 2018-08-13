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

from rl_coach.filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from rl_coach.spaces import ActionSpace, DiscreteActionSpace


class FullDiscreteActionSpaceMap(PartialDiscreteActionSpaceMap):
    """
    Maps all the actions in the output space to discrete actions in the action space.
    For example, if there are 10 multiselect actions in the output space, the actions 0-9 will be mapped to those
    multiselect actions.
    """
    def __init__(self):
        super().__init__()

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> DiscreteActionSpace:
        self.target_actions = output_action_space.actions
        return super().get_unfiltered_action_space(output_action_space)
