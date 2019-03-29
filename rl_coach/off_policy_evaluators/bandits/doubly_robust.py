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


class DoublyRobust(object):

    @staticmethod
    def evaluate(ope_shared_stats: 'OpeSharedStats') -> tuple:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        Papers:
        https://arxiv.org/abs/1103.4601
        https://arxiv.org/pdf/1612.01205 (some more clearer explanations)

        :return: the evaluation score
        """

        ips = np.mean(ope_shared_stats.rho_all_dataset * ope_shared_stats.all_rewards)
        dm = np.mean(ope_shared_stats.all_v_values_reward_model_based)
        dr = np.mean(ope_shared_stats.rho_all_dataset *
                     (ope_shared_stats.all_rewards - ope_shared_stats.all_reward_model_rewards[
                         range(len(ope_shared_stats.all_actions)), ope_shared_stats.all_actions])) + dm

        return ips, dm, dr
