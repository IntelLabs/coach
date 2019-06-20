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

import copy
from typing import Union
from collections import OrderedDict

import numpy as np

from rl_coach.agents.agent import Agent
from rl_coach.agents.ddpg_agent import DDPGAgent
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import DDPGActorHeadParameters, TD3VHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters, EmbedderScheme
from rl_coach.core_types import ActionInfo, TrainingSteps, Transition
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import BoxActionSpace, GoalsSpace


class TD3CriticNetworkParameters(NetworkParameters):
    def __init__(self, num_q_networks):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(),
                                            'action': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)}
        self.middleware_parameters = FCMiddlewareParameters(num_streams=num_q_networks)
        self.heads_parameters = [TD3VHeadParameters()]
        self.optimizer_type = 'Adam'
        self.adam_optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-8
        self.batch_size = 100
        self.async_training = False
        self.learning_rate = 0.001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class TD3ActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [DDPGActorHeadParameters(batchnorm=False)]
        self.optimizer_type = 'Adam'
        self.adam_optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-8
        self.batch_size = 100
        self.async_training = False
        self.learning_rate = 0.001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class TD3AlgorithmParameters(AlgorithmParameters):
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
        self.rate_for_copying_weights_to_target = 0.005
        self.use_target_network_for_evaluation = False
        self.action_penalty = 0
        self.clip_critic_targets = None  # expected to be a tuple of the form (min_clip_value, max_clip_value) or None
        self.use_non_zero_discount_for_terminal_states = False
        self.act_for_full_episodes = True
        self.update_policy_every_x_episode_steps = 2
        self.num_steps_between_copying_online_weights_to_target = TrainingSteps(self.update_policy_every_x_episode_steps)
        self.policy_noise = 0.2
        self.noise_clipping = 0.5
        self.num_q_networks = 2


class TD3AgentExplorationParameters(AdditiveNoiseParameters):
    def __init__(self):
        super().__init__()
        self.noise_as_percentage_from_action_space = False


class TD3AgentParameters(AgentParameters):
    def __init__(self):
        td3_algorithm_params = TD3AlgorithmParameters()
        super().__init__(algorithm=td3_algorithm_params,
                         exploration=TD3AgentExplorationParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks=OrderedDict([("actor", TD3ActorNetworkParameters()),
                                               ("critic",
                                                TD3CriticNetworkParameters(td3_algorithm_params.num_q_networks))]))

    @property
    def path(self):
        return 'rl_coach.agents.td3_agent:TD3Agent'


# Twin Delayed DDPG - https://arxiv.org/pdf/1802.09477.pdf
class TD3Agent(DDPGAgent):
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

        # add noise to the next_actions
        noise = np.random.normal(0, self.ap.algorithm.policy_noise, next_actions.shape).clip(
            -self.ap.algorithm.noise_clipping, self.ap.algorithm.noise_clipping)
        next_actions = self.spaces.action.clip_action_to_space(next_actions + noise)

        critic_inputs = copy.copy(batch.next_states(critic_keys))
        critic_inputs['action'] = next_actions
        q_st_plus_1 = critic.target_network.predict(critic_inputs)[2]  # output #2 is the min (Q1, Q2)

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

        # train the critic
        critic_inputs = copy.copy(batch.states(critic_keys))
        critic_inputs['action'] = batch.actions(len(batch.actions().shape) == 1)
        result = critic.train_and_sync_networks(critic_inputs, TD_targets)
        total_loss, losses, unclipped_grads = result[:3]

        if self.training_iteration % self.ap.algorithm.update_policy_every_x_episode_steps == 0:
            # get the gradients of output #3 (=mean of Q1 network) w.r.t the action
            critic_inputs = copy.copy(batch.states(critic_keys))
            critic_inputs['action'] = actions_mean
            action_gradients = critic.online_network.predict(critic_inputs,
                                                             outputs=critic.online_network.gradients_wrt_inputs[3]['action'])

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
        self.ap.algorithm.num_consecutive_training_steps = self.current_episode_steps_counter
        return Agent.train(self)

    def update_transition_before_adding_to_replay_buffer(self, transition: Transition) -> Transition:
        """
        Allows agents to update the transition just before adding it to the replay buffer.
        Can be useful for agents that want to tweak the reward, termination signal, etc.

        :param transition: the transition to update
        :return: the updated transition
        """
        transition.game_over = False if self.current_episode_steps_counter ==\
                                        self.parent_level_manager.environment.env._max_episode_steps\
            else transition.game_over

        return transition