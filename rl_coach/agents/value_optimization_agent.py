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
        # TODO continue with restructring the code below. We assume that we have an episodic ER. We then iterate in batches
        #  on all the transitions in the episodes, and for each transition, we update in its 'info':
        #  1. DM (Reward Model) reward prediction - this can be done only once right after training the reward network.
        #  2. Trained policy Q value
        #  3. Trained Policy Softmax based prob
        #  4. Trained policy V value
        #
        #  we then gather all of the above metrics in variables containing this data for all of the transitions in the
        #  dataset to calculate IPS, DM, and DR.
        #  we can then have all of the needed metrics all sorted out for SDR (especially the state value).


        # IPS
        if self.first:
            import tensorflow as tf
            self.a = tf.placeholder(tf.float32, shape=(None, len(self.spaces.action.actions)))
            self.prob = tf.nn.softmax(self.a, axis=1)

            self.first_state = deepcopy(self.memory.get_episode(0).transitions[0].state)
            self.first_state['observation'] = np.expand_dims(self.first_state['observation'], axis=0)
            self.first = False

        def softmax(x, temperature):
            """Compute softmax values for each sets of scores in x."""
            x = x / temperature
            return self.networks['main'].sess.run(self.prob, feed_dict={self.a: x})

        all_learned_policy_probs_temp_0_2 = np.empty((0, len(self.spaces.action.actions)))
        all_learned_policy_probs_temp_0_5 = np.empty((0, len(self.spaces.action.actions)))
        all_learned_policy_probs_temp_0_75 = np.empty((0, len(self.spaces.action.actions)))
        all_learned_policy_probs_temp_2 = np.empty((0, len(self.spaces.action.actions)))
        all_learned_policy_probs = np.empty((0, len(self.spaces.action.actions)))

        all_q_values = np.empty((0, len(self.spaces.action.actions)))
        all_rewards = np.empty((0))
        all_reward_model_rewards = np.empty((0, len(self.spaces.action.actions)))
        all_actions = np.empty((0)).astype('int')
        network_parameters = self.ap.network_wrappers['main']
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # generate model probabilities
        # TODO this should use the evaluation dataset, and not the training dataset
        for batch in self.call_memory('get_shuffled_data_generator', network_parameters.batch_size):
            batch = Batch(batch)
            # TODO: just get all the rewards from the ER directly. Set batch size to the size of the ER maybe? or otherwise.
            all_rewards = np.concatenate([all_rewards, batch.rewards()])
            all_reward_model_rewards = \
                np.concatenate([all_reward_model_rewards, self.networks['reward_model'].online_network.
                               predict(batch.states(network_keys))], axis=0)

            q_values = self.networks['main'].online_network.predict(batch.states(network_keys))
            all_q_values = np.concatenate([all_q_values, q_values], axis=0)
            all_learned_policy_probs = np.concatenate([all_learned_policy_probs, softmax(q_values, 0.35)], axis=0)
            all_learned_policy_probs_temp_0_2 = np.concatenate([all_learned_policy_probs_temp_0_2, softmax(q_values, 0.2)], axis=0)
            all_learned_policy_probs_temp_0_5 = np.concatenate([all_learned_policy_probs_temp_0_5, softmax(q_values, 0.5)], axis=0)
            all_learned_policy_probs_temp_2 = np.concatenate([all_learned_policy_probs_temp_2, softmax(q_values, 2)], axis=0)
            all_actions = np.concatenate([all_actions, batch.actions()])

        old_policy_prob = 0.5
        new_policy_prob = np.max(all_learned_policy_probs, axis=1)
        rho_all_dataset = new_policy_prob/old_policy_prob

        IPS = np.mean(rho_all_dataset * all_rewards)

        all_learned_policy_state_values = np.sum(all_learned_policy_probs * all_reward_model_rewards, axis=1)
        DM = np.mean(all_learned_policy_state_values)

        DR = np.mean(rho_all_dataset *
                     (all_rewards - all_reward_model_rewards[range(len(all_actions)), all_actions])) + DM

        print("Q_Values and Probabilities")
        print("==========================")
        print("Q_Values: " + str([q for q in list(all_q_values[0:10])]))
        print("Probs temp = 0.2 " + str([p for p in list(all_learned_policy_probs_temp_0_2[0:10])]))
        print("Probs temp = 0.35 " + str([p for p in list(all_learned_policy_probs[0:10])]))
        print("Probs temp = 0.5 " + str([p for p in list(all_learned_policy_probs_temp_0_5[0:10])]))
        print("Probs temp = 0.75 " + str([p for p in list(all_learned_policy_probs_temp_0_75[0:10])]))
        print("Probs temp = 2 " + str([p for p in list(all_learned_policy_probs_temp_2[0:10])]))

        print("Bandits")
        print("=======")
        print("IPS Estimator = {}".format(IPS))
        print("DM Estimator = {}".format(DM))
        print("DR Estimator = {}".format(DR))

        print("DQN_Agent - testing for Q value magnitude: Q_value for first state, first episode = {}".
              format(self.networks['main'].online_network.predict(self.first_state)))

        # Sequential Doubly Robust
        episodes = [self.call_memory('get_episode', i) for i in range(self.call_memory('num_complete_episodes'))]
        for episode in episodes:
            for transition in episode.transitions:
                # per_episode_seq_dr =
                pass

    def improve_reward_model(self):
        batch_size = self.ap.network_wrappers['reward_model'].batch_size
        network_keys = self.ap.network_wrappers['reward_model'].input_embedders_parameters.keys()

        # this is fitted from the training dataset, as does the policy
        # TODO extract hyper-param out
        # 100 epochs should be enough to learn some reasonable model
        for epoch in range(100):
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






