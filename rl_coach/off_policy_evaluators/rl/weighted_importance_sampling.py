#
# Copyright (c) 2019 Intel Corporation
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

from rl_coach.core_types import Episode


class WeightedImportanceSampling(object):
# TODO rename and add PDIS
    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: List[Episode]) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        References:
        - Sutton, R. S. & Barto, A. G. Reinforcement Learning: An Introduction. Chapter 5.5.
        - https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
        - http://videolectures.net/deeplearning2017_thomas_safe_rl/

        :return: the evaluation score
        """

        # Weighted Importance Sampling
        per_episode_w_i = []

        for episode in evaluation_dataset_as_episodes:
            w_i = 1
            for transition in episode.transitions:
                w_i *= transition.info['softmax_policy_prob'][transition.action] / \
                      transition.info['all_action_probabilities'][transition.action]
            per_episode_w_i.append(w_i)

        total_w_i_sum_across_episodes = sum(per_episode_w_i)
        wis = 0
        for i, episode in enumerate(evaluation_dataset_as_episodes):
            wis += per_episode_w_i[i]/total_w_i_sum_across_episodes * episode.transitions[0].n_step_discounted_rewards

        return wis
