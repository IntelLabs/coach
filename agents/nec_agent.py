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

import numpy as np

from agents.value_optimization_agent import *


# Neural Episodic Control - https://arxiv.org/pdf/1703.01988.pdf
class NECAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id,
                                        create_target_network=False)
        self.current_episode_state_embeddings = []
        self.current_episode_actions = []
        self.training_started = False
        # if self.tp.checkpoint_restore_dir:
        #     self.load_dnd(self.tp.checkpoint_restore_dir)

    def learn_from_batch(self, batch):
        if not self.main_network.online_network.output_heads[0].DND.has_enough_entries(self.tp.agent.number_of_knn):
            return 0
        else:
            if not self.training_started:
                self.training_started = True
                screen.log_title("Finished collecting initial entries in DND. Starting to train network...")

        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)
        result = self.main_network.train_and_sync_networks([current_states, actions], total_return)
        total_loss = result[0]

        return total_loss

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        """
        this method modifies the superclass's behavior in only 3 ways:

        1) the embedding is saved and stored in self.current_episode_state_embeddings
        2) the dnd output head is only called if it has a minimum number of entries in it
            ideally, the dnd had would do this on its own, but in my attempt in encoding this
            behavior in tensorflow, I ran into problems. Would definitely be worth
            revisiting in the future
        3) during training, actions are saved and stored in self.current_episode_actions
            if behaviors 1 and 2 were handled elsewhere, this could easily be implemented
            as a wrapper around super instead of overriding this method entirelysearch
        """

        # get embedding
        embedding = self.main_network.online_network.predict(
            self.tf_input_state(curr_state),
            outputs=self.main_network.online_network.state_embedding)
        self.current_episode_state_embeddings.append(embedding)

        # TODO: support additional heads.  Right now all other heads are ignored
        if self.main_network.online_network.output_heads[0].DND.has_enough_entries(self.tp.agent.number_of_knn):
            # if there are enough entries in the DND then we can query it to get the action values
            # actions_q_values = []
            feed_dict = {
                self.main_network.online_network.state_embedding: [embedding],
            }
            actions_q_values = self.main_network.sess.run(
                self.main_network.online_network.output_heads[0].output, feed_dict=feed_dict)
        else:
            # get only the embedding so we can insert it to the DND
            actions_q_values = [0] * self.action_space_size

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        if phase == RunPhase.TRAIN:
            action = self.exploration_policy.get_action(actions_q_values)
            # NOTE: this next line is not in the parent implementation
            # NOTE: it could be implemented as a wrapper around the parent since action is returned
            self.current_episode_actions.append(action)
        else:
            action = np.argmax(actions_q_values)

        # store the q values statistics for logging
        self.q_values.add_sample(actions_q_values)

        # store information for plotting interactively (actual plotting is done in agent)
        if self.tp.visualization.plot_action_values_online:
            for idx, action_name in enumerate(self.env.actions_description):
                self.episode_running_info[action_name].append(actions_q_values[idx])

        action_value = {"action_value": actions_q_values[action]}
        return action, action_value

    def reset_game(self, do_not_reset_env=False):
        ValueOptimizationAgent.reset_game(self, do_not_reset_env)

        # make sure we already have at least one episode
        if self.memory.num_complete_episodes() >= 1 and not self.in_heatup:
            # get the last full episode that we have collected
            episode = self.memory.get(-2)
            returns = []
            for i in range(episode.length()):
                returns.append(episode.get_transition(i).total_return)
            # Just to deal with the end of heatup where there might be a case where it ends in a middle
            # of an episode, and thus when getting the episode out of the ER, it will be a complete one whereas
            # the other statistics collected here, are collected only during training.
            returns = returns[-len(self.current_episode_actions):]
            self.main_network.online_network.output_heads[0].DND.add(self.current_episode_state_embeddings,
                                                                     self.current_episode_actions, returns)

        self.current_episode_state_embeddings = []
        self.current_episode_actions = []

    def save_model(self, model_id):
        self.main_network.save_model(model_id)
        with open(os.path.join(self.tp.save_model_dir, str(model_id) + '.dnd'), 'wb') as f:
            pickle.dump(self.main_network.online_network.output_heads[0].DND, f, pickle.HIGHEST_PROTOCOL)
