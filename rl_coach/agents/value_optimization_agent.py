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
        self.first = True

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

    def get_prediction(self, states):
        return self.networks['main'].online_network.predict(self.prepare_batch_for_inference(states, 'main'))

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
            actions_q_values = actions_q_values.squeeze()

            # store the q values statistics for logging
            self.q_values.add_sample(actions_q_values)
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

    def run_ope(self):
        network_parameters = self.ap.network_wrappers['main']
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()
        batch_size = network_parameters.batch_size

        # IPS
        def softmax(x, temperature):
            """Compute softmax values for each sets of scores in x."""
            x = x / temperature
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        all_reward_model_rewards, all_q_values, all_policy_probs, all_old_policy_probs = [], [], [], []
        all_v_values_reward_model_based, all_v_values_q_model_based, all_rewards, all_actions = [], [], [], []

        transitions = self.call_memory('get_all_full_episodes_transitions')
        for i in range(int(len(transitions) / batch_size) + 1):
            batch = transitions[i * batch_size: (i + 1) * batch_size]
            batch_for_inference = Batch(batch)

            all_reward_model_rewards.append(self.networks['reward_model'].online_network.predict(
                batch_for_inference.states(network_keys)))
            all_q_values.append(self.networks['main'].online_network.predict(batch_for_inference.states(network_keys)))
            all_policy_probs.append(softmax(all_q_values[-1], 0.35))
            all_v_values_reward_model_based.append(np.sum(all_policy_probs[-1] * all_reward_model_rewards[-1], axis=1))
            all_v_values_q_model_based.append(np.sum(all_policy_probs[-1] * all_q_values[-1], axis=1))
            all_rewards.append(batch_for_inference.rewards())
            all_actions.append(batch_for_inference.actions())
            all_old_policy_probs.append(batch_for_inference.info('action_probability'))

            for j, t in enumerate(batch):
                t.update_info({
                    'model_reward': all_reward_model_rewards[-1][j],
                    'q_value': all_q_values[-1][j],
                    'softmax_policy_prob': all_policy_probs[-1][j],
                    'v_value_q_model_based': all_v_values_q_model_based[-1][j],

                })

        all_reward_model_rewards = np.concatenate(all_reward_model_rewards, axis=0)
        all_q_values = np.concatenate(all_q_values, axis=0)
        all_policy_probs = np.concatenate(all_policy_probs, axis=0)
        all_v_values_reward_model_based = np.concatenate(all_v_values_reward_model_based, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_old_policy_probs = np.concatenate(all_old_policy_probs, axis=0)

        # generate model probabilities
        # TODO this should use the evaluation dataset, and not the training dataset

        new_policy_prob = np.max(all_policy_probs, axis=1)
        rho_all_dataset = new_policy_prob/all_old_policy_probs

        IPS = np.mean(rho_all_dataset * all_rewards)
        DM = np.mean(all_v_values_reward_model_based)
        DR = np.mean(rho_all_dataset *
                     (all_rewards - all_reward_model_rewards[range(len(all_actions)), all_actions])) + DM

        print("Q_Values: {} \n".format(str([q for q in list(all_q_values[0])])))

        print("Bandits")
        print("=======")
        print("IPS Estimator = {}".format(IPS))
        print("DM Estimator = {}".format(DM))
        print("DR Estimator = {}".format(DR))

        # Sequential Doubly Robust
        episodes = [self.call_memory('get_episode', i) for i in range(self.call_memory('num_complete_episodes'))]
        per_episode_seq_dr = []

        for episode in episodes:
            episode_seq_dr = 0
            for transition in episode.transitions:
                rho = transition.info['softmax_policy_prob'][transition.action] / transition.info['action_probability']
                episode_seq_dr = transition.info['v_value_q_model_based'] + rho * (transition.reward + self.ap.algorithm.discount
                                                                         * episode_seq_dr -
                                                                         transition.info['q_value'][transition.action])
            per_episode_seq_dr.append(episode_seq_dr)

        SEQ_DR = np.array(per_episode_seq_dr).mean()

        print("RL")
        print("=======")
        print("SEQ_DR Estimator = {}".format(SEQ_DR))

    def improve_reward_model(self):
        batch_size = self.ap.network_wrappers['reward_model'].batch_size
        network_keys = self.ap.network_wrappers['reward_model'].input_embedders_parameters.keys()

        # this is fitted from the training dataset, as does the policy
        # TODO extract hyper-param out
        # 100 epochs should be enough to learn some reasonable model
        for epoch in range(50):
            log = OrderedDict()
            log['Epoch'] = epoch
            screen.log_dict(log, prefix='Training Reward Model')
            for batch in self.call_memory('get_shuffled_data_generator', batch_size):
                batch = Batch(batch)
                current_rewards_prediction_for_all_actions = self.networks['reward_model'].online_network.predict(batch.states(network_keys))
                current_rewards_prediction_for_all_actions[range(batch_size), batch.actions()] = batch.rewards()
                loss = self.networks['reward_model'].train_and_sync_networks(batch.states(network_keys),
                                                                             current_rewards_prediction_for_all_actions)[0]
                # print("epoch {}: loss = {}".format(epoch, loss))
        # print(self.networks['reward_model'].online_network.predict(batch.states(network_keys)))






