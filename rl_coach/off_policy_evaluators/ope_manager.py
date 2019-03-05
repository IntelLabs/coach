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
from typing import List

from rl_coach.architectures.architecture import Architecture
from rl_coach.core_types import Episode, Batch
from rl_coach.off_policy_evaluators.bandits.doubly_robust import DoublyRobust
from rl_coach.off_policy_evaluators.rl.sequential_doubly_robust import SequentialDoublyRobust

from rl_coach.core_types import Transition


class OpeManager(object):
    def __init__(self):
        self.doubly_robust = DoublyRobust()
        self.sequential_doubly_robust = SequentialDoublyRobust()


    @staticmethod
    def _prepare_shared_stats(dataset_as_transitions: List[Transition], batch_size: int,
                              reward_model: Architecture, q_network: Architecture, network_keys: List,
                              temperature: float):
        # IPS
        # TODO have softmax calculated as part of the Q network
        def softmax(x, temperature):
            """Compute softmax values for each sets of scores in x."""
            x = x / temperature
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        all_reward_model_rewards, all_policy_probs, all_old_policy_probs = [], [], []
        all_v_values_reward_model_based, all_v_values_q_model_based, all_rewards, all_actions = [], [], [], []

        for i in range(int(len(dataset_as_transitions) / batch_size) + 1):
            batch = dataset_as_transitions[i * batch_size: (i + 1) * batch_size]
            batch_for_inference = Batch(batch)

            all_reward_model_rewards.append(reward_model.predict(
                batch_for_inference.states(network_keys)))
            q_values = q_network.predict(batch_for_inference.states(network_keys))
            all_policy_probs.append(softmax(q_values, temperature))
            all_v_values_reward_model_based.append(np.sum(all_policy_probs[-1] * all_reward_model_rewards[-1], axis=1))
            all_v_values_q_model_based.append(np.sum(all_policy_probs[-1] * q_values, axis=1))
            all_rewards.append(batch_for_inference.rewards())
            all_actions.append(batch_for_inference.actions())
            all_old_policy_probs.append(batch_for_inference.info('all_action_probabilities')
                                        [range(len(batch_for_inference.actions())), batch_for_inference.actions()])

            for j, t in enumerate(batch):
                t.update_info({
                    'q_value': q_values[j],
                    'softmax_policy_prob': all_policy_probs[-1][j],
                    'v_value_q_model_based': all_v_values_q_model_based[-1][j],

                })

            # DEBUG
            if i == 0:
                print("Q_Values: {} \n".format(str([q for q in list(q_values[0])])))

        all_reward_model_rewards = np.concatenate(all_reward_model_rewards, axis=0)
        all_policy_probs = np.concatenate(all_policy_probs, axis=0)
        all_v_values_reward_model_based = np.concatenate(all_v_values_reward_model_based, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_old_policy_probs = np.concatenate(all_old_policy_probs, axis=0)

        # generate model probabilities
        new_policy_prob = np.max(all_policy_probs, axis=1)
        rho_all_dataset = new_policy_prob / all_old_policy_probs

        # TODO these should all go into some container class to put some order in here. maybe a namedtuple.
        return all_reward_model_rewards, all_policy_probs, all_v_values_reward_model_based, all_rewards, all_actions, \
               all_old_policy_probs, new_policy_prob, rho_all_dataset

    def evaluate(self, dataset_as_episodes: List[Episode], batch_size: int, discount_factor: float,
                 reward_model: Architecture, q_network: Architecture, network_keys: List):
        """

        :param dataset_as_episodes:
        :param batch_size:
        :param discount_factor:
        :param reward_model:
        :param q_network:
        :param network_keys:
        :return:
        """
        # TODO this should use the evaluation dataset, and not the training dataset

        dataset_as_transitions = [t for e in dataset_as_episodes for t in e.transitions]
        temperature = 0.2

        all_reward_model_rewards, all_policy_probs, all_v_values_reward_model_based, all_rewards, all_actions, \
        all_old_policy_probs, new_policy_prob, rho_all_dataset = self._prepare_shared_stats(
            dataset_as_transitions, batch_size, reward_model, q_network, network_keys, temperature)

        ips, dm, dr = self.doubly_robust.evaluate(rho_all_dataset, all_rewards, all_v_values_reward_model_based,
                                    all_reward_model_rewards, all_actions)
        seq_dr = self.sequential_doubly_robust.evaluate(dataset_as_episodes, temperature, discount_factor)

        return ips, dm, dr, seq_dr

