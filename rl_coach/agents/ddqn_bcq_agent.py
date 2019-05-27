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
from collections import OrderedDict

from copy import deepcopy
from typing import Union, List, Dict
import numpy as np

from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters, DQNAgent
from rl_coach.base_parameters import Parameters
from rl_coach.core_types import EnvironmentSteps, Batch, StateType
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.logger import screen
from rl_coach.memories.non_episodic.differentiable_neural_dictionary import AnnoyDictionary
from rl_coach.schedules import LinearSchedule


class NNImitationModelParameters(Parameters):
    """
    A parameters module grouping together parameters related to a neural network based action selection.
    """
    def __init__(self):
        super().__init__()
        self.imitation_model_num_epochs = 100
        self.mask_out_actions_threshold = 0.35


class KNNParameters(Parameters):
    """
    A parameters module grouping together parameters related to a k-Nearest Neighbor based action selection.
    """
    def __init__(self):
        super().__init__()
        self.average_dist_coefficient = 1
        self.knn_size = 50000
        self.use_state_embedding_instead_of_state = True  # useful when the state is too big to be used for kNN


class DDQNBCQAlgorithmParameters(DQNAlgorithmParameters):
    """
    :param action_drop_method_parameters: (Parameters)
        Defines the mode and related parameters according to which low confidence actions will be filtered out
    :param num_steps_between_copying_online_weights_to_target (StepMethod)
        Defines the number of steps between every phase of copying online network's weights to the target network's weights
    """
    def __init__(self):
        super().__init__()
        self.action_drop_method_parameters = KNNParameters()
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

        if isinstance(self.ap.algorithm.action_drop_method_parameters, KNNParameters):
            self.knn_trees = []  # will be filled out later, as we don't have the action space size yet
            self.average_dist = 0

            def to_embedding(states: Union[List[StateType], Dict]):
                if isinstance(states, list):
                    states = self.prepare_batch_for_inference(states, 'reward_model')
                if self.ap.algorithm.action_drop_method_parameters.use_state_embedding_instead_of_state:
                    return self.networks['reward_model'].online_network.predict(
                        states,
                        outputs=[self.networks['reward_model'].online_network.state_embedding])
                else:
                    return states['observation']
            self.embedding = to_embedding

        elif isinstance(self.ap.algorithm.action_drop_method_parameters, NNImitationModelParameters):
            if 'imitation_model' not in self.ap.network_wrappers:
                # user hasn't defined params for the reward model. we will use the same params as used for the 'main'
                # network.
                self.ap.network_wrappers['imitation_model'] = deepcopy(self.ap.network_wrappers['reward_model'])
        else:
            raise ValueError('Unsupported action drop method {} for DDQNBCQAgent'.format(
                type(self.ap.algorithm.action_drop_method_parameters)))

    def select_actions(self, next_states, q_st_plus_1):
        if isinstance(self.ap.algorithm.action_drop_method_parameters, KNNParameters):
            familiarity = np.array([[distance[0] for distance in
                            knn_tree._get_k_nearest_neighbors_indices(self.embedding(next_states), 1)[0]]
                                    for knn_tree in self.knn_trees]).transpose()
            actions_to_mask_out = familiarity > self.ap.algorithm.action_drop_method_parameters.average_dist_coefficient \
                                  * self.average_dist

        elif isinstance(self.ap.algorithm.action_drop_method_parameters, NNImitationModelParameters):
            familiarity = self.networks['imitation_model'].online_network.predict(next_states)
            actions_to_mask_out = familiarity < \
                                    self.ap.algorithm.action_drop_method_parameters.mask_out_actions_threshold
        else:
            raise ValueError('Unsupported action drop method {} for DDQNBCQAgent'.format(
                type(self.ap.algorithm.action_drop_method_parameters)))

        masked_next_q_values = self.networks['main'].online_network.predict(next_states)
        masked_next_q_values[actions_to_mask_out] = -np.inf

        # occassionaly there are states in the batch for which our model shows no confidence for either of the actions
        # in that case, we will just randomly assign q_values to actions, since otherwise argmax will always return
        # the first action
        zero_confidence_rows = (masked_next_q_values.max(axis=1) == -np.inf)
        masked_next_q_values[zero_confidence_rows] = np.random.rand(np.sum(zero_confidence_rows),
                                                                    masked_next_q_values.shape[1])

        return np.argmax(masked_next_q_values, 1)

    def improve_reward_model(self, epochs: int):
        """
        Train both a reward model to be used by the doubly-robust estimator, and some model to be used for BCQ

        :param epochs: The total number of epochs to use for training a reward model
        :return: None
        """

        # we'll be assuming that these gets drawn from the reward model parameters
        batch_size = self.ap.network_wrappers['reward_model'].batch_size
        network_keys = self.ap.network_wrappers['reward_model'].input_embedders_parameters.keys()

        # if using a NN to decide which actions to drop, we'll train the NN here
        if isinstance(self.ap.algorithm.action_drop_method_parameters, NNImitationModelParameters):
            total_epochs = max(epochs, self.ap.algorithm.action_drop_method_parameters.imitation_model_num_epochs)
        else:
            total_epochs = epochs

        for epoch in range(total_epochs):
            # this is fitted from the training dataset
            reward_model_loss = 0
            imitation_model_loss = 0
            total_transitions_processed = 0
            for i, batch in enumerate(self.call_memory('get_shuffled_training_data_generator', batch_size)):
                batch = Batch(batch)

                # reward model
                if epoch < epochs:
                    reward_model_loss += self.get_reward_model_loss(batch)

                # imitation model
                if isinstance(self.ap.algorithm.action_drop_method_parameters, NNImitationModelParameters) and \
                        epoch < self.ap.algorithm.action_drop_method_parameters.imitation_model_num_epochs:
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

        # if using a kNN based model, we'll initialize and build it here.
        # initialization cannot be moved to the constructor as we don't have the agent's spaces initialized yet.
        if isinstance(self.ap.algorithm.action_drop_method_parameters, KNNParameters):
            knn_size = self.ap.algorithm.action_drop_method_parameters.knn_size
            if self.ap.algorithm.action_drop_method_parameters.use_state_embedding_instead_of_state:
                self.knn_trees = [AnnoyDictionary(
                    dict_size=knn_size,
                    key_width=int(self.networks['reward_model'].online_network.state_embedding.shape[-1]),
                    batch_size=knn_size)
                    for _ in range(len(self.spaces.action.actions))]
            else:
                self.knn_trees = [AnnoyDictionary(
                    dict_size=knn_size,
                    key_width=self.spaces.state['observation'].shape[0],
                    batch_size=knn_size)
                    for _ in range(len(self.spaces.action.actions))]

            for i, knn_tree in enumerate(self.knn_trees):
                state_embeddings = self.embedding([transition.state for transition in self.memory.transitions
                                if transition.action == i])
                knn_tree.add(
                    keys=state_embeddings,
                    values=np.expand_dims(np.zeros(state_embeddings.shape[0]), axis=1))

            for knn_tree in self.knn_trees:
                knn_tree._rebuild_index()

            self.average_dist = [[dist[0] for dist in knn_tree._get_k_nearest_neighbors_indices(
                keys=self.embedding([transition.state for transition in self.memory.transitions]),
                k=1)[0]] for knn_tree in self.knn_trees]
            self.average_dist = sum([x for l in self.average_dist for x in l])  # flatten and sum
            self.average_dist /= len(self.memory.transitions)

    def set_session(self, sess) -> None:
        super().set_session(sess)

        # we check here if we are in batch-rl, since this is the only place where we have a graph_manager to question
        assert isinstance(self.parent_level_manager.parent_graph_manager, BatchRLGraphManager),\
            'DDQNBCQ agent can only be used in BatchRL'
