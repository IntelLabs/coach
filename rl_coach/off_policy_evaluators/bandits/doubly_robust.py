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
import numpy as np

from rl_coach.off_policy_evaluators.off_policy_evaluator import OffPolicyEvaluator


class DoublyRobust(OffPolicyEvaluator):
    def __init__(self):
        """
        A doubly robust estimator, based on <paper_link>. This estimator is aimed to evaluate policies for
        bandits problems.
        """
        pass

    def evaluate(self, rho_all_dataset, all_rewards, all_v_values_reward_model_based,
                 all_reward_model_rewards, all_actions) -> tuple:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        :return: the evaluation score
        """

        ips = np.mean(rho_all_dataset * all_rewards)
        dm = np.mean(all_v_values_reward_model_based)
        dr = np.mean(rho_all_dataset *
                     (all_rewards - all_reward_model_rewards[range(len(all_actions)), all_actions])) + dm

        print("Bandits")
        print("=======")
        print("IPS Estimator = {}".format(ips))
        print("DM Estimator = {}".format(dm))
        print("DR Estimator = {}".format(dr))

        return ips, dm, dr