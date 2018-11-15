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

from rl_coach.exploration_policies.additive_noise import AdditiveNoise, AdditiveNoiseParameters


class ContinuousEntropyParameters(AdditiveNoiseParameters):
    @property
    def path(self):
        return 'rl_coach.exploration_policies.continuous_entropy:ContinuousEntropy'


class ContinuousEntropy(AdditiveNoise):
    """
    Continuous entropy is an exploration policy that is actually implemented as part of the network.
    The exploration policy class is only a placeholder for choosing this policy. The exploration policy is
    implemented by adding a regularization factor to the network loss, which regularizes the entropy of the action.
    This exploration policy is only intended for continuous action spaces, and assumes that the entire calculation
    is implemented as part of the head.

    .. warning::
       This exploration policy expects the agent or the network to implement the exploration functionality.
       There are only a few heads that actually are relevant and implement the entropy regularization factor.
    """
    pass
