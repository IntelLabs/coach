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
import math
from collections import namedtuple

import numpy as np
from typing import List

from rl_coach.architectures.architecture import Architecture
from rl_coach.core_types import Episode, Batch
from rl_coach.off_policy_evaluators.bandits.doubly_robust import DoublyRobust
from rl_coach.off_policy_evaluators.rl.sequential_doubly_robust import SequentialDoublyRobust

from rl_coach.core_types import Transition

from rl_coach.off_policy_evaluators.rl.weighted_importance_sampling import WeightedImportanceSampling

OpeSharedStats = namedtuple("OpeSharedStats", ['all_reward_model_rewards', 'all_policy_probs',
                                               'all_v_values_reward_model_based', 'all_rewards', 'all_actions',
                                               'all_old_policy_probs', 'new_policy_prob', 'rho_all_dataset'])
OpeEstimation = namedtuple("OpeEstimation", ['ips', 'dm', 'dr', 'seq_dr', 'wis'])


class OpeManager(object):
    def __init__(self):
        self.evaluation_dataset_as_transitions = None
        self.doubly_robust = DoublyRobust()
        self.sequential_doubly_robust = SequentialDoublyRobust()
        self.weighted_importance_sampling = WeightedImportanceSampling()
        self.all_reward_model_rewards = None
        self.all_old_policy_probs = None
        self.all_rewards = None
        self.all_actions = None
        self.is_gathered_static_shared_data = False

    def _prepare_ope_shared_stats(self, evaluation_dataset_as_transitions: List[Transition], batch_size: int,
                                  q_network: Architecture, network_keys: List) -> OpeSharedStats:
        """
        Do the preparations needed for different estimators.
        Some of the calcuations are shared, so we centralize all the work here.

        :param evaluation_dataset_as_transitions: The evaluation dataset in the form of transitions.
        :param batch_size: The batch size to use.
        :param reward_model: A reward model to be used by DR
        :param q_network: The Q network whose its policy we evaluate.
        :param network_keys: The network keys used for feeding the neural networks.
        :return:
        """

        assert self.is_gathered_static_shared_data, "gather_static_shared_stats() should be called once before " \
                                                    "calling _prepare_ope_shared_stats()"
        # IPS
        all_policy_probs = []
        all_v_values_reward_model_based, all_v_values_q_model_based = [], []

        for i in range(math.ceil(len(evaluation_dataset_as_transitions) / batch_size)):
            batch = evaluation_dataset_as_transitions[i * batch_size: (i + 1) * batch_size]
            batch_for_inference = Batch(batch)

            # we always use the first Q head to calculate OPEs. might want to change this in the future.
            # for instance, this means that for bootstrapped dqn we always use the first QHead to calculate the OPEs.
            q_values, sm_values = q_network.predict(batch_for_inference.states(network_keys),
                                                    outputs=[q_network.output_heads[0].q_values,
                                                             q_network.output_heads[0].softmax])

            all_policy_probs.append(sm_values)
            all_v_values_reward_model_based.append(np.sum(all_policy_probs[-1] * self.all_reward_model_rewards[i],
                                                          axis=1))
            all_v_values_q_model_based.append(np.sum(all_policy_probs[-1] * q_values, axis=1))

            for j, t in enumerate(batch):
                t.update_info({
                    'q_value': q_values[j],
                    'softmax_policy_prob': all_policy_probs[-1][j],
                    'v_value_q_model_based': all_v_values_q_model_based[-1][j],

                })

        all_policy_probs = np.concatenate(all_policy_probs, axis=0)
        all_v_values_reward_model_based = np.concatenate(all_v_values_reward_model_based, axis=0)

        # generate model probabilities
        new_policy_prob = all_policy_probs[np.arange(self.all_actions.shape[0]), self.all_actions]
        rho_all_dataset = new_policy_prob / self.all_old_policy_probs

        return OpeSharedStats(self.all_reward_model_rewards, all_policy_probs, all_v_values_reward_model_based,
                              self.all_rewards, self.all_actions, self.all_old_policy_probs, new_policy_prob,
                              rho_all_dataset)

    def gather_static_shared_stats(self, evaluation_dataset_as_transitions: List[Transition], batch_size: int,
                                   reward_model: Architecture, network_keys: List) -> None:
        all_reward_model_rewards = []
        all_old_policy_probs = []
        all_rewards = []
        all_actions = []

        for i in range(math.ceil(len(evaluation_dataset_as_transitions) / batch_size)):
            batch = evaluation_dataset_as_transitions[i * batch_size: (i + 1) * batch_size]
            batch_for_inference = Batch(batch)

            all_reward_model_rewards.append(reward_model.predict(batch_for_inference.states(network_keys)))
            all_rewards.append(batch_for_inference.rewards())
            all_actions.append(batch_for_inference.actions())
            all_old_policy_probs.append(batch_for_inference.info('all_action_probabilities')
                                             [range(len(batch_for_inference.actions())),
                                              batch_for_inference.actions()])

        self.all_reward_model_rewards = np.concatenate(all_reward_model_rewards, axis=0)
        self.all_old_policy_probs = np.concatenate(all_old_policy_probs, axis=0)
        self.all_rewards = np.concatenate(all_rewards, axis=0)
        self.all_actions = np.concatenate(all_actions, axis=0)

        # mark that static shared data was collected and ready to be used
        self.is_gathered_static_shared_data = True

    def evaluate(self, evaluation_dataset_as_episodes: List[Episode], evaluation_dataset_as_transitions: List[Transition], batch_size: int,
                 discount_factor: float, q_network: Architecture, network_keys: List) -> OpeEstimation:
        """
        Run all the OPEs and get estimations of the current policy performance based on the evaluation dataset.

        :param evaluation_dataset_as_episodes: The evaluation dataset in a form of episodes.
        :param evaluation_dataset_as_transitions: The evaluation dataset in a form of transitions.
        :param batch_size: Batch size to use for the estimators.
        :param discount_factor: The standard RL discount factor.
        :param reward_model: A reward model to be used by DR
        :param q_network: The Q network whose its policy we evaluate.
        :param network_keys: The network keys used for feeding the neural networks.

        :return: An OpeEstimation tuple which groups together all the OPE estimations
        """
        ope_shared_stats = self._prepare_ope_shared_stats(evaluation_dataset_as_transitions, batch_size, q_network,
                                                          network_keys)

        ips, dm, dr = self.doubly_robust.evaluate(ope_shared_stats)
        seq_dr = self.sequential_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
        wis = self.weighted_importance_sampling.evaluate(evaluation_dataset_as_episodes)
        
        return OpeEstimation(ips, dm, dr, seq_dr, wis)

