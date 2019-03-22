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

import copy
from enum import Enum
from typing import Union

import numpy as np

from rl_coach.agents.agent import Agent
from rl_coach.architectures.head_parameters import MeasurementsPredictionHeadParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.tensorflow_components.layers import Conv2d, Dense
from rl_coach.base_parameters import AlgorithmParameters, AgentParameters, NetworkParameters, \
     MiddlewareScheme
from rl_coach.core_types import ActionInfo, EnvironmentSteps, RunPhase
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.spaces import SpacesDefinition, VectorObservationSpace


class HandlingTargetsAfterEpisodeEnd(Enum):
    LastStep = 0
    NAN = 1


class DFPNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='leaky_relu'),
                                            'measurements': InputEmbedderParameters(activation_function='leaky_relu'),
                                            'goal': InputEmbedderParameters(activation_function='leaky_relu')}

        self.input_embedders_parameters['observation'].scheme = [
            Conv2d(32, 8, 4),
            Conv2d(64, 4, 2),
            Conv2d(64, 3, 1),
            Dense(512),
        ]

        self.input_embedders_parameters['measurements'].scheme = [
            Dense(128),
            Dense(128),
            Dense(128),
        ]

        self.input_embedders_parameters['goal'].scheme = [
            Dense(128),
            Dense(128),
            Dense(128),
        ]

        self.middleware_parameters = FCMiddlewareParameters(activation_function='leaky_relu',
                                                            scheme=MiddlewareScheme.Empty)
        self.heads_parameters = [MeasurementsPredictionHeadParameters(activation_function='leaky_relu')]
        self.async_training = False
        self.batch_size = 64
        self.adam_optimizer_beta1 = 0.95


class DFPMemoryParameters(EpisodicExperienceReplayParameters):
    def __init__(self):
        self.max_size = (MemoryGranularity.Transitions, 20000)
        self.shared_memory = True
        super().__init__()


class DFPAlgorithmParameters(AlgorithmParameters):
    """
    :param num_predicted_steps_ahead: (int)
        Number of future steps to predict measurements for. The future steps won't be sequential, but rather jump
        in multiples of 2. For example, if num_predicted_steps_ahead = 3, then the steps will be: t+1, t+2, t+4.
        The predicted steps will be [t + 2**i for i in range(num_predicted_steps_ahead)]

    :param goal_vector: (List[float])
        The goal vector will weight each of the measurements to form an optimization goal. The vector should have
        the same length as the number of measurements, and it will be vector multiplied by the measurements.
        Positive values correspond to trying to maximize the particular measurement, and negative values
        correspond to trying to minimize the particular measurement.

    :param future_measurements_weights: (List[float])
        The future_measurements_weights weight the contribution of each of the predicted timesteps to the optimization
        goal. For example, if there are 6 steps predicted ahead, and a future_measurements_weights vector with 3 values,
        then only the 3 last timesteps will be taken into account, according to the weights in the
        future_measurements_weights vector.

    :param use_accumulated_reward_as_measurement: (bool)
        If set to True, the accumulated reward from the beginning of the episode will be added as a measurement to
        the measurements vector in the state. This van be useful in environments where the given measurements don't
        include enough information for the particular goal the agent should achieve.

    :param handling_targets_after_episode_end: (HandlingTargetsAfterEpisodeEnd)
        Dictates how to handle measurements that are outside the episode length.

    :param scale_measurements_targets: (Dict[str, float])
        Allows rescaling the values of each of the measurements available. This van be useful when the measurements
        have a different scale and you want to normalize them to the same scale.
    """
    def __init__(self):
        super().__init__()
        self.num_predicted_steps_ahead = 6
        self.goal_vector = [1.0, 1.0]
        self.future_measurements_weights = [0.5, 0.5, 1.0]
        self.use_accumulated_reward_as_measurement = False
        self.handling_targets_after_episode_end = HandlingTargetsAfterEpisodeEnd.NAN
        self.scale_measurements_targets = {}
        self.num_consecutive_playing_steps = EnvironmentSteps(8)


class DFPAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DFPAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=DFPMemoryParameters(),
                         networks={"main": DFPNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.dfp_agent:DFPAgent'


# Direct Future Prediction Agent - http://vladlen.info/papers/learning-to-act.pdf
class DFPAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.current_goal = self.ap.algorithm.goal_vector
        self.target_measurements_scale_factors = None

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        network_inputs = batch.states(network_keys)
        network_inputs['goal'] = np.repeat(np.expand_dims(self.current_goal, 0),
                                           batch.size, axis=0)

        # get the current outputs of the network
        targets = self.networks['main'].online_network.predict(network_inputs)

        # change the targets for the taken actions
        for i in range(batch.size):
            targets[i, batch.actions()[i]] = batch[i].info['future_measurements'].flatten()

        result = self.networks['main'].train_and_sync_networks(network_inputs, targets)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

    def choose_action(self, curr_state):
        if self.exploration_policy.requires_action_values():
            # predict the future measurements
            tf_input_state = self.prepare_batch_for_inference(curr_state, 'main')
            tf_input_state['goal'] = np.expand_dims(self.current_goal, 0)
            measurements_future_prediction = self.networks['main'].online_network.predict(tf_input_state)[0]
            action_values = np.zeros(len(self.spaces.action.actions))
            num_steps_used_for_objective = len(self.ap.algorithm.future_measurements_weights)

            # calculate the score of each action by multiplying it's future measurements with the goal vector
            for action_idx in range(len(self.spaces.action.actions)):
                action_measurements = measurements_future_prediction[action_idx]
                action_measurements = np.reshape(action_measurements,
                                                 (self.ap.algorithm.num_predicted_steps_ahead,
                                                  self.spaces.state['measurements'].shape[0]))
                future_steps_values = np.dot(action_measurements, self.current_goal)
                action_values[action_idx] = np.dot(future_steps_values[-num_steps_used_for_objective:],
                                                   self.ap.algorithm.future_measurements_weights)
        else:
            action_values = None

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        action = self.exploration_policy.get_action(action_values)

        if action_values is not None:
            action_values = action_values.squeeze()
            action_info = ActionInfo(action=action, action_value=action_values[action])
        else:
            action_info = ActionInfo(action=action)

        return action_info

    def set_environment_parameters(self, spaces: SpacesDefinition):
        self.spaces = copy.deepcopy(spaces)
        self.spaces.goal = VectorObservationSpace(shape=self.spaces.state['measurements'].shape,
                                                  measurements_names=
                                                  self.spaces.state['measurements'].measurements_names)

        # if the user has filled some scale values, check that he got the names right
        if set(self.spaces.state['measurements'].measurements_names).intersection(
                self.ap.algorithm.scale_measurements_targets.keys()) !=\
                set(self.ap.algorithm.scale_measurements_targets.keys()):
            raise ValueError("Some of the keys in parameter scale_measurements_targets ({})  are not defined in "
                             "the measurements space {}".format(self.ap.algorithm.scale_measurements_targets.keys(),
                                                                self.spaces.state['measurements'].measurements_names))

        super().set_environment_parameters(self.spaces)

        # the below is done after calling the base class method, as it might add accumulated reward as a measurement

        # fill out the missing measurements scale factors
        for measurement_name in self.spaces.state['measurements'].measurements_names:
            if measurement_name not in self.ap.algorithm.scale_measurements_targets:
                self.ap.algorithm.scale_measurements_targets[measurement_name] = 1

        self.target_measurements_scale_factors = \
            np.array([self.ap.algorithm.scale_measurements_targets[measurement_name] for measurement_name in
                      self.spaces.state['measurements'].measurements_names])

    def handle_episode_ended(self):
        last_episode = self.current_episode_buffer
        if self.phase in [RunPhase.TRAIN, RunPhase.HEATUP] and last_episode:
            self._update_measurements_targets(last_episode,
                                              self.ap.algorithm.num_predicted_steps_ahead)
        super().handle_episode_ended()

    def _update_measurements_targets(self, episode, num_steps):
        if 'measurements' not in episode.transitions[0].state or episode.transitions[0].state['measurements'] == []:
            raise ValueError("Measurements are not present in the transitions of the last episode played. ")
        measurements_size = self.spaces.state['measurements'].shape[0]
        for transition_idx, transition in enumerate(episode.transitions):
            transition.info['future_measurements'] = np.zeros((num_steps, measurements_size))
            for step in range(num_steps):
                offset_idx = transition_idx + 2 ** step

                if offset_idx >= episode.length():
                    if self.ap.algorithm.handling_targets_after_episode_end == HandlingTargetsAfterEpisodeEnd.NAN:
                        # the special MSE loss will ignore those entries so that the gradient will be 0 for these
                        transition.info['future_measurements'][step] = np.nan
                        continue

                    elif self.ap.algorithm.handling_targets_after_episode_end == HandlingTargetsAfterEpisodeEnd.LastStep:
                        offset_idx = - 1

                transition.info['future_measurements'][step] = \
                    self.target_measurements_scale_factors * \
                    (episode.transitions[offset_idx].state['measurements'] - transition.state['measurements'])
