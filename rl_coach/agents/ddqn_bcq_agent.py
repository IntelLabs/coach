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
from copy import deepcopy
from typing import Union
import numpy as np

from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters, DQNAgent
from rl_coach.core_types import EnvironmentSteps, Batch
from rl_coach.logger import screen
from rl_coach.schedules import LinearSchedule


class DDQNBCQAlgorithmParameters(DQNAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.imitation_model_num_epochs = 100
        self.mask_out_actions_threshold = 0.35
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(30000)


class DDQNBCQAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm = DDQNBCQAlgorithmParameters()
        self.exploration.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
        self.exploration.evaluation_epsilon = 0.001

    @property
    def path(self):
        return 'rl_coach.agents.ddqn_bcq_agent:DDQNBCQAgent'


# Double DQN - https://arxiv.org/abs/1509.06461
# (a variant on) BCQ - https://arxiv.org/pdf/1812.02900v2.pdf
class DDQNBCQAgent(DQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        if 'imitation_model' not in self.ap.network_wrappers:
            # user hasn't defined params for the reward model. we will use the same params as used for the 'main'
            # network.
            self.ap.network_wrappers['imitation_model'] = deepcopy(self.ap.network_wrappers['reward_model'])

    def select_actions(self, next_states, q_st_plus_1):
        familiarity = self.networks['imitation_model'].online_network.predict(next_states)
        actions_to_mask_out = familiarity < self.ap.algorithm.mask_out_actions_threshold
        masked_next_q_values = self.networks['main'].online_network.predict(next_states)
        masked_next_q_values[actions_to_mask_out] = -np.inf

        return np.argmax(masked_next_q_values, 1)

    def improve_reward_model(self, epochs: int):
        """
        Train both a reward model to be used by the doubly-robust estimator, and a imitation model to be used for BCQ

        :param epochs: The total number of epochs to use for training a reward model
        :return: None
        """

        # we'll be assuming that these gets drawn from the reward model parameters
        batch_size = self.ap.network_wrappers['reward_model'].batch_size
        network_keys = self.ap.network_wrappers['reward_model'].input_embedders_parameters.keys()

        total_epochs = max(epochs, self.ap.algorithm.imitation_model_num_epochs)
        for epoch in range(total_epochs):
            # this is fitted from the training dataset
            reward_model_loss = 0
            imitation_model_loss = 0
            total_transitions_processed = 0
            for i, batch in enumerate(self.call_memory('get_shuffled_data_generator', batch_size)):
                batch = Batch(batch)

                # reward model
                if epoch < epochs:
                    current_rewards_prediction_for_all_actions = self.networks['reward_model'].online_network.predict(
                        batch.states(network_keys))
                    current_rewards_prediction_for_all_actions[range(batch.size), batch.actions()] = batch.rewards()
                    reward_model_loss += self.networks['reward_model'].train_and_sync_networks(
                        batch.states(network_keys), current_rewards_prediction_for_all_actions)[0]

                # imitation model
                if epoch < self.ap.algorithm.imitation_model_num_epochs:
                    target_actions = np.zeros((batch.size, len(self.spaces.action.actions)))
                    target_actions[range(batch.size), batch.actions()] = 1
                    imitation_model_loss += self.networks['imitation_model'].train_and_sync_networks(
                        batch.states(network_keys), target_actions)[0]

                total_transitions_processed += batch.size

            log = OrderedDict()
            log['Epoch'] = epoch

            if reward_model_loss:
                log['Reward Model Loss'] = reward_model_loss / total_transitions_processed
            if imitation_model_loss:
                log['Imitation Model Loss'] = imitation_model_loss / total_transitions_processed

            screen.log_dict(log, prefix='Training Batch RL Models')

# TODO instead of a NN, try using several AnnoyDictionary and manage those directly to do a NN search for embeddings to the state.
#  Try doing it both for embeddings and to the state directly. Can maybe use the reward model embedding as an input query to the NN tree.