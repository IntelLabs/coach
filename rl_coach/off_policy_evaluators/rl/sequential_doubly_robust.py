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
from rl_coach.off_policy_evaluators.off_policy_evaluator import OffPolicyEvaluator


class SequentialDoublyRobust(OffPolicyEvaluator):
    def __init__(self):
        """
        A sequential doubly robust estimator, based on <paper_link>. This estimator is aimed to evaluate policies for
        reinforcement learning problems.
        """
        pass

    def evaluate(self, dataset_as_episodes: List[Episode], temperature, discount_factor) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        :return: the evaluation score
        """

        # Sequential Doubly Robust
        per_episode_seq_dr = []

        for episode in dataset_as_episodes:
            episode_seq_dr = 0
            for transition in episode.transitions:
                rho = transition.info['softmax_policy_prob'][transition.action] / \
                      transition.info['all_action_probabilities'][transition.action]
                episode_seq_dr = transition.info['v_value_q_model_based'] + rho * (transition.reward + discount_factor
                                                                         * episode_seq_dr -
                                                                         transition.info['q_value'][transition.action])
            per_episode_seq_dr.append(episode_seq_dr)

        SEQ_DR = np.array(per_episode_seq_dr).mean()

        print("RL")
        print("=======")
        print("Temperature = {}: SEQ_DR Estimator = {}".format(temperature, SEQ_DR))

        return SEQ_DR
