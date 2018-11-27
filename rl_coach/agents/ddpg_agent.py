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
from typing import Union
from collections import OrderedDict

import numpy as np

from rl_coach.agents.actor_critic_agent import ActorCriticAgent
from rl_coach.agents.agent import Agent
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import DDPGActorHeadParameters, VHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters, EmbedderScheme
from rl_coach.core_types import ActionInfo, EnvironmentSteps
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import BoxActionSpace, GoalsSpace


class DDPGCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True),
                                            'action': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class DDPGActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True)}
        self.middleware_parameters = FCMiddlewareParameters(batchnorm=True)
        self.heads_parameters = [DDPGActorHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.0001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class DDPGAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_copying_online_weights_to_target: (StepMethod)
        The number of steps between copying the online network weights to the target network weights.

    :param rate_for_copying_weights_to_target: (float)
        When copying the online network weights to the target network weights, a soft update will be used, which
        weight the new online network weights by rate_for_copying_weights_to_target

    :param num_consecutive_playing_steps: (StepMethod)
        The number of consecutive steps to act between every two training iterations

    :param use_target_network_for_evaluation: (bool)
        If set to True, the target network will be used for predicting the actions when choosing actions to act.
        Since the target network weights change more slowly, the predicted actions will be more consistent.

    :param action_penalty: (float)
        The amount by which to penalize the network on high action feature (pre-activation) values.
        This can prevent the actions features from saturating the TanH activation function, and therefore prevent the
        gradients from becoming very low.

    :param clip_critic_targets: (Tuple[float, float] or None)
        The range to clip the critic target to in order to prevent overestimation of the action values.

    :param use_non_zero_discount_for_terminal_states: (bool)
        If set to True, the discount factor will be used for terminal states to bootstrap the next predicted state
        values. If set to False, the terminal states reward will be taken as the target return for the network.
    """
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.001
        self.num_consecutive_playing_steps = EnvironmentSteps(1)
        self.use_target_network_for_evaluation = False
        self.action_penalty = 0
        self.clip_critic_targets = None  # expected to be a tuple of the form (min_clip_value, max_clip_value) or None
        self.use_non_zero_discount_for_terminal_states = False


class DDPGAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DDPGAlgorithmParameters(),
                         exploration=OUProcessParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks=OrderedDict([("actor", DDPGActorNetworkParameters()),
                                               ("critic", DDPGCriticNetworkParameters())]))

    @property
    def path(self):
        return 'rl_coach.agents.ddpg_agent:DDPGAgent'


# Deep Deterministic Policy Gradients Network - https://arxiv.org/pdf/1509.02971.pdf
class DDPGAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

        self.q_values = self.register_signal("Q")
        self.TD_targets_signal = self.register_signal("TD targets")
        self.action_signal = self.register_signal("actions")

    def learn_from_batch(self, batch):
        actor = self.networks['actor']
        critic = self.networks['critic']

        actor_keys = self.ap.network_wrappers['actor'].input_embedders_parameters.keys()
        critic_keys = self.ap.network_wrappers['critic'].input_embedders_parameters.keys()

        # TD error = r + discount*max(q_st_plus_1) - q_st
        next_actions, actions_mean = actor.parallel_prediction([
            (actor.target_network, batch.next_states(actor_keys)),
            (actor.online_network, batch.states(actor_keys))
        ])

        critic_inputs = copy.copy(batch.next_states(critic_keys))
        critic_inputs['action'] = next_actions
        q_st_plus_1 = critic.target_network.predict(critic_inputs)

        # calculate the bootstrapped TD targets while discounting terminal states according to
        # use_non_zero_discount_for_terminal_states
        if self.ap.algorithm.use_non_zero_discount_for_terminal_states:
            TD_targets = batch.rewards(expand_dims=True) + self.ap.algorithm.discount * q_st_plus_1
        else:
            TD_targets = batch.rewards(expand_dims=True) + \
                         (1.0 - batch.game_overs(expand_dims=True)) * self.ap.algorithm.discount * q_st_plus_1

        # clip the TD targets to prevent overestimation errors
        if self.ap.algorithm.clip_critic_targets:
            TD_targets = np.clip(TD_targets, *self.ap.algorithm.clip_critic_targets)

        self.TD_targets_signal.add_sample(TD_targets)

        # get the gradients of the critic output with respect to the action
        critic_inputs = copy.copy(batch.states(critic_keys))
        critic_inputs['action'] = actions_mean
        action_gradients = critic.online_network.predict(critic_inputs,
                                                         outputs=critic.online_network.gradients_wrt_inputs[0]['action'])

        # train the critic
        critic_inputs = copy.copy(batch.states(critic_keys))
        critic_inputs['action'] = batch.actions(len(batch.actions().shape) == 1)
        result = critic.train_and_sync_networks(critic_inputs, TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        # apply the gradients from the critic to the actor
        initial_feed_dict = {actor.online_network.gradients_weights_ph[0]: -action_gradients}
        gradients = actor.online_network.predict(batch.states(actor_keys),
                                                 outputs=actor.online_network.weighted_gradients[0],
                                                 initial_feed_dict=initial_feed_dict)

        if actor.has_global:
            actor.apply_gradients_to_global_network(gradients)
            actor.update_online_network()
        else:
            actor.apply_gradients_to_online_network(gradients)

        return total_loss, losses, unclipped_grads

    def train(self):
        return Agent.train(self)

    def choose_action(self, curr_state):
        if not (isinstance(self.spaces.action, BoxActionSpace) or isinstance(self.spaces.action, GoalsSpace)):
            raise ValueError("DDPG works only for continuous control problems")
        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'actor')
        if self.ap.algorithm.use_target_network_for_evaluation:
            actor_network = self.networks['actor'].target_network
        else:
            actor_network = self.networks['actor'].online_network

        action_values = actor_network.predict(tf_input_state).squeeze()

        action = self.exploration_policy.get_action(action_values)

        self.action_signal.add_sample(action)

        # get q value
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'critic')
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        tf_input_state['action'] = action_batch
        q_value = self.networks['critic'].online_network.predict(tf_input_state)[0]
        self.q_values.add_sample(q_value)

        action_info = ActionInfo(action=action,
                                 action_value=q_value)

        return action_info
