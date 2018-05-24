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
import os, pickle
from agents.value_optimization_agent import ValueOptimizationAgent
from logger import screen
from utils import RunPhase


# Neural Episodic Control - https://arxiv.org/pdf/1703.01988.pdf
class NECAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id,
                                        create_target_network=False)
        self.current_episode_state_embeddings = []
        self.training_started = False

    def learn_from_batch(self, batch):
        if not self.main_network.online_network.output_heads[0].DND.has_enough_entries(self.tp.agent.number_of_knn):
            return 0
        else:
            if not self.training_started:
                self.training_started = True
                screen.log_title("Finished collecting initial entries in DND. Starting to train network...")

        current_states, next_states, actions, rewards, game_overs, total_return = self.extract_batch(batch)

        TD_targets = self.main_network.online_network.predict(current_states)

        #  only update the action that we have actually done in this transition
        for i in range(self.tp.batch_size):
            TD_targets[i, actions[i]] = total_return[i]

        # train the neural network
        result = self.main_network.train_and_sync_networks(current_states, TD_targets)

        total_loss = result[0]

        return total_loss

    def act(self, phase=RunPhase.TRAIN):
        if self.in_heatup:
            # get embedding in heatup (otherwise we get it through choose_action)
            embedding = self.main_network.online_network.predict(
                self.tf_input_state(self.curr_state),
                outputs=self.main_network.online_network.state_embedding)
            self.current_episode_state_embeddings.append(embedding)

        return super().act(phase)

    def get_prediction(self, curr_state):
        # get the actions q values and the state embedding
        embedding, actions_q_values = self.main_network.online_network.predict(
            self.tf_input_state(curr_state),
            outputs=[self.main_network.online_network.state_embedding,
                     self.main_network.online_network.output_heads[0].output]
        )

        # store the state embedding for inserting it to the DND later
        self.current_episode_state_embeddings.append(embedding.squeeze())
        actions_q_values = actions_q_values[0][0]
        return actions_q_values

    def reset_game(self, do_not_reset_env=False):
        super().reset_game(do_not_reset_env)

        # get the last full episode that we have collected
        episode = self.memory.get_last_complete_episode()
        if episode is not None:
            # the indexing is only necessary because the heatup can end in the middle of an episode
            # this won't be required after fixing this so that when the heatup is ended, the episode is closed
            returns = episode.get_transitions_attribute('total_return')[:len(self.current_episode_state_embeddings)]
            actions = episode.get_transitions_attribute('action')[:len(self.current_episode_state_embeddings)]
            self.main_network.online_network.output_heads[0].DND.add(self.current_episode_state_embeddings,
                                                                     actions, returns)

        self.current_episode_state_embeddings = []

    def save_model(self, model_id):
        self.main_network.save_model(model_id)
        with open(os.path.join(self.tp.save_model_dir, str(model_id) + '.dnd'), 'wb') as f:
            pickle.dump(self.main_network.online_network.output_heads[0].DND, f, pickle.HIGHEST_PROTOCOL)
