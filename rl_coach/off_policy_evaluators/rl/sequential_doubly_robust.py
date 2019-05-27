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


class SequentialDoublyRobust(object):

    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: List[Episode], discount_factor: float) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).
        When the epsiodes are of changing lengths, this estimator might prove problematic due to its nature of recursion
        of adding rewards up to the end of the episode (horizon). It will probably work best with episodes of fixed
        length.
        Paper: https://arxiv.org/pdf/1511.03722.pdf

        :return: the evaluation score
        """

        # Sequential Doubly Robust
        per_episode_seq_dr = []

        for episode in evaluation_dataset_as_episodes:
            episode_seq_dr = 0
            for transition in reversed(episode.transitions):
                rho = transition.info['softmax_policy_prob'][transition.action] / \
                      transition.info['all_action_probabilities'][transition.action]
                episode_seq_dr = transition.info['v_value_q_model_based'] + rho * (transition.reward + discount_factor
                                                                                   * episode_seq_dr -
                                                                                   transition.info['q_value'][
                                                                                       transition.action])
            per_episode_seq_dr.append(episode_seq_dr)

        seq_dr = np.array(per_episode_seq_dr).mean()

        return seq_dr
