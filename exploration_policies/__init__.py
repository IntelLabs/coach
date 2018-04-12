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
from exploration_policies.additive_noise import AdditiveNoise
from exploration_policies.approximated_thompson_sampling_using_dropout import ApproximatedThompsonSamplingUsingDropout
from exploration_policies.bayesian import Bayesian
from exploration_policies.boltzmann import Boltzmann
from exploration_policies.bootstrapped import Bootstrapped
from exploration_policies.categorical import Categorical
from exploration_policies.continuous_entropy import ContinuousEntropy
from exploration_policies.e_greedy import EGreedy
from exploration_policies.exploration_policy import ExplorationPolicy
from exploration_policies.greedy import Greedy
from exploration_policies.ou_process import OUProcess
from exploration_policies.thompson_sampling import ThompsonSampling


__all__ = [AdditiveNoise,
           ApproximatedThompsonSamplingUsingDropout,
           Bayesian,
           Boltzmann,
           Bootstrapped,
           Categorical,
           ContinuousEntropy,
           EGreedy,
           ExplorationPolicy,
           Greedy,
           OUProcess,
           ThompsonSampling]
