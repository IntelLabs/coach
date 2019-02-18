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

from typing import Union

import numpy as np

from rl_coach.agents.agent import Agent
from rl_coach.core_types import ActionInfo, StateType, Batch
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplay
from rl_coach.spaces import DiscreteActionSpace


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
        network_parameters = list(self.ap.network_wrappers.values())[0]

        # IPS
        import tensorflow as tf
        a = tf.placeholder(tf.float32, shape=(None, len(self.spaces.action.actions)))
        prob = tf.nn.softmax(a, axis=1)
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        def softmax(x, temperature):
            """Compute softmax values for each sets of scores in x."""
            x = x / temperature
            return self.networks['main'].sess.run(prob, feed_dict={a: x})

        all_probs = np.empty((0, 2))
        all_rewards = np.empty((0, 1))
        # generate model probabilities
        for batch in self.call_memory('get_shuffled_data_generator', network_parameters.batch_size):
            batch = Batch(batch)
            all_rewards = np.concatenate([all_rewards, batch.rewards(expand_dims=True)])
            q_values = self.networks['main'].online_network.predict(batch.states(network_keys))
            all_probs = np.concatenate([all_probs, softmax(q_values, 0.35)], axis=0)

        old_policy_prob = 0.5
        new_policy_prob = np.max(all_probs, axis=1)
        print(np.mean(new_policy_prob/old_policy_prob * all_rewards[0]))





