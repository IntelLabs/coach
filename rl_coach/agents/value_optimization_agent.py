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
from collections import OrderedDict
from typing import Union

import numpy as np

from rl_coach.agents.agent import Agent
from rl_coach.core_types import ActionInfo, StateType, Batch
from rl_coach.filters.filter import NoInputFilter
from rl_coach.logger import screen
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplay
from rl_coach.spaces import DiscreteActionSpace
from copy import deepcopy

## This is an abstract agent - there is no learn_from_batch method ##


class ValueOptimizationAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.q_values = self.register_signal("Q")
        self.q_value_for_action = {}

    def init_environment_dependent_modules(self):
        super().init_environment_dependent_modules()
        if isinstance(self.spaces.action, DiscreteActionSpace):
            for i in range(len(self.spaces.action.actions)):
                self.q_value_for_action[i] = self.register_signal("Q for action {}".format(i),
                                                                  dump_one_value_per_episode=False,
                                                                  dump_one_value_per_step=True)

    # Algorithms for which q_values are calculated from predictions will override this function
    def get_all_q_values_for_states(self, states: StateType):
        if self.exploration_policy.requires_action_values():
            actions_q_values = self.get_prediction(states)
        else:
            actions_q_values = None
        return actions_q_values

    def get_prediction(self, states, outputs=None):
        return self.networks['main'].online_network.predict(self.prepare_batch_for_inference(states, 'main'),
                                                            outputs=outputs)

    def update_transition_priorities_and_get_weights(self, TD_errors, batch):
        # update errors in prioritized replay buffer
        importance_weights = None
        if isinstance(self.memory, PrioritizedExperienceReplay):
            self.call_memory('update_priorities', (batch.info('idx'), TD_errors))
            importance_weights = batch.info('weight')
        return importance_weights

    def _validate_action(self, policy, action):
        if np.array(action).shape != ():
            raise ValueError((
                'The exploration_policy {} returned a vector of actions '
                'instead of a single action. ValueOptimizationAgents '
                'require exploration policies which return a single action.'
            ).format(policy.__class__.__name__))

    def choose_action(self, curr_state):
        actions_q_values = self.get_all_q_values_for_states(curr_state)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        action = self.exploration_policy.get_action(actions_q_values)
        self._validate_action(self.exploration_policy, action)

        if actions_q_values is not None:
            # this is for bootstrapped dqn
            if type(actions_q_values) == list and len(actions_q_values) > 0:
                actions_q_values = self.exploration_policy.last_action_values

            # store the q values statistics for logging
            self.q_values.add_sample(actions_q_values)

            actions_q_values = actions_q_values.squeeze()

            for i, q_value in enumerate(actions_q_values):
                self.q_value_for_action[i].add_sample(q_value)

            action_info = ActionInfo(action=action,
                                     action_value=actions_q_values[action],
                                     max_action_value=np.max(actions_q_values))
        else:
            action_info = ActionInfo(action=action)

        return action_info

    def learn_from_batch(self, batch):
        raise NotImplementedError("ValueOptimizationAgent is an abstract agent. Not to be used directly.")

    def run_off_policy_evaluation(self):
        """
        Run the off-policy evaluation estimators to get a prediction for the performance of the current policy based on
        an evaluation dataset, which was collected by another policy(ies).
        :return: None
        """
        assert self.ope_manager

        if not isinstance(self.pre_network_filter, NoInputFilter) and len(self.pre_network_filter.reward_filters) != 0:
            raise ValueError("Defining a pre-network reward filter when OPEs are calculated will result in a mismatch "
                             "between q values (which are scaled), and actual rewards, which are not. It is advisable "
                             "to use an input_filter, if possible, instead, which will filter the transitions directly "
                             "in the replay buffer, affecting both the q_values and the rewards themselves. ")

        ips, dm, dr, seq_dr, wis = self.ope_manager.evaluate(
                                  evaluation_dataset_as_episodes=self.memory.evaluation_dataset_as_episodes,
                                  evaluation_dataset_as_transitions=self.memory.evaluation_dataset_as_transitions,
                                  batch_size=self.ap.network_wrappers['main'].batch_size,
                                  discount_factor=self.ap.algorithm.discount,
                                  q_network=self.networks['main'].online_network,
                                  network_keys=list(self.ap.network_wrappers['main'].input_embedders_parameters.keys()))

        # get the estimators out to the screen
        log = OrderedDict()
        log['Epoch'] = self.training_epoch
        log['IPS'] = ips
        log['DM'] = dm
        log['DR'] = dr
        log['WIS'] = wis
        log['Sequential-DR'] = seq_dr
        screen.log_dict(log, prefix='Off-Policy Evaluation')

        # get the estimators out to dashboard
        self.agent_logger.set_current_time(self.get_current_time() + 1)
        self.agent_logger.create_signal_value('Inverse Propensity Score', ips)
        self.agent_logger.create_signal_value('Direct Method Reward', dm)
        self.agent_logger.create_signal_value('Doubly Robust', dr)
        self.agent_logger.create_signal_value('Sequential Doubly Robust', seq_dr)
        self.agent_logger.create_signal_value('Weighted Importance Sampling', wis)

    def get_reward_model_loss(self, batch: Batch):
        network_keys = self.ap.network_wrappers['reward_model'].input_embedders_parameters.keys()
        current_rewards_prediction_for_all_actions = self.networks['reward_model'].online_network.predict(
            batch.states(network_keys))
        current_rewards_prediction_for_all_actions[range(batch.size), batch.actions()] = batch.rewards()

        return self.networks['reward_model'].train_and_sync_networks(
            batch.states(network_keys), current_rewards_prediction_for_all_actions)[0]

    def improve_reward_model(self, epochs: int):
        """
        Train a reward model to be used by the doubly-robust estimator

        :param epochs: The total number of epochs to use for training a reward model
        :return: None
        """
        batch_size = self.ap.network_wrappers['reward_model'].batch_size

        # this is fitted from the training dataset
        for epoch in range(epochs):
            loss = 0
            total_transitions_processed = 0
            for i, batch in enumerate(self.call_memory('get_shuffled_training_data_generator', batch_size)):
                batch = Batch(batch)
                loss += self.get_reward_model_loss(batch)
                total_transitions_processed += batch.size

            log = OrderedDict()
            log['Epoch'] = epoch
            log['loss'] = loss / total_transitions_processed
            screen.log_dict(log, prefix='Training Reward Model')

