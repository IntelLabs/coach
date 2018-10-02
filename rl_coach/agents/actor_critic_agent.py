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
import scipy.signal

from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent, PolicyGradientRescaler
from rl_coach.architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from rl_coach.architectures.tensorflow_components.heads.v_head import VHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, \
    AgentParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.exploration_policies.continuous_entropy import ContinuousEntropyParameters
from rl_coach.logger import screen
from rl_coach.memories.episodic.single_episode_buffer import SingleEpisodeBufferParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace
from rl_coach.utils import last_sample
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters


class ActorCriticAlgorithmParameters(AlgorithmParameters):
    """
    :param policy_gradient_rescaler: (PolicyGradientRescaler)
    The value that will be used to rescale the policy gradient

    :param apply_gradients_every_x_episodes: (int)
    The number of episodes to wait before applying the accumulated gradients to the network.
    The training iterations only accumulate gradients without actually applying them.

    :param beta_entropy: (float)
    The weight that will be given to the entropy regularization which is used in order to improve exploration.

    :param num_steps_between_gradient_updates: (int)
    Every num_steps_between_gradient_updates transitions will be considered as a single batch and use for
    accumulating gradients. This is also the number of steps used for bootstrapping according to the n-step formulation.

    :param gae_lambda: (float)
    If the policy gradient rescaler was defined as PolicyGradientRescaler.GAE, the generalized advantage estimation
    scheme will be used, in which case the lambda value controls the decay for the different n-step lengths.

    :param estimate_state_value_using_gae: (bool)
    If set to True, the state value targets for the V head will be estimated using the GAE scheme.
    """
    def __init__(self):
        super().__init__()
        self.policy_gradient_rescaler = PolicyGradientRescaler.A_VALUE
        self.apply_gradients_every_x_episodes = 5
        self.beta_entropy = 0
        self.num_steps_between_gradient_updates = 5000  # this is called t_max in all the papers
        self.gae_lambda = 0.96
        self.estimate_state_value_using_gae = False


class ActorCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters(loss_weight=0.5), PolicyHeadParameters(loss_weight=1.0)]
        self.optimizer_type = 'Adam'
        self.clip_gradients = 40.0
        self.async_training = True


class ActorCriticAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=ActorCriticAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: ContinuousEntropyParameters()},
                         memory=SingleEpisodeBufferParameters(),
                         networks={"main": ActorCriticNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.actor_critic_agent:ActorCriticAgent'


# Actor Critic - https://arxiv.org/abs/1602.01783
class ActorCriticAgent(PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.last_gradient_update_step_idx = 0
        self.action_advantages = self.register_signal('Advantages')
        self.state_values = self.register_signal('Values')
        self.value_loss = self.register_signal('Value Loss')
        self.policy_loss = self.register_signal('Policy Loss')

    # Discounting function used to calculate discounted returns.
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def get_general_advantage_estimation_values(self, rewards, values):
        # values contain n+1 elements (t ... t+n+1), rewards contain n elements (t ... t + n)
        bootstrap_extended_rewards = np.array(rewards.tolist() + [values[-1]])

        # Approximation based calculation of GAE (mathematically correct only when Tmax = inf,
        # although in practice works even in much smaller Tmax values, e.g. 20)
        deltas = rewards + self.ap.algorithm.discount * values[1:] - values[:-1]
        gae = self.discount(deltas, self.ap.algorithm.discount * self.ap.algorithm.gae_lambda)

        if self.ap.algorithm.estimate_state_value_using_gae:
            discounted_returns = np.expand_dims(gae + values[:-1], -1)
        else:
            discounted_returns = np.expand_dims(np.array(self.discount(bootstrap_extended_rewards,
                                                                       self.ap.algorithm.discount)), 1)[:-1]
        return gae, discounted_returns

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # get the values for the current states

        result = self.networks['main'].online_network.predict(batch.states(network_keys))
        current_state_values = result[0]

        self.state_values.add_sample(current_state_values)

        # the targets for the state value estimator
        num_transitions = batch.size
        state_value_head_targets = np.zeros((num_transitions, 1))

        # estimate the advantage function
        action_advantages = np.zeros((num_transitions, 1))

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            if batch.game_overs()[-1]:
                R = 0
            else:
                R = self.networks['main'].online_network.predict(last_sample(batch.next_states(network_keys)))[0]

            for i in reversed(range(num_transitions)):
                R = batch.rewards()[i] + self.ap.algorithm.discount * R
                state_value_head_targets[i] = R
                action_advantages[i] = R - current_state_values[i]

        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            bootstrapped_value = self.networks['main'].online_network.predict(last_sample(batch.next_states(network_keys)))[0]
            values = np.append(current_state_values, bootstrapped_value)
            if batch.game_overs()[-1]:
                values[-1] = 0

            # get general discounted returns table
            gae_values, state_value_head_targets = self.get_general_advantage_estimation_values(batch.rewards(), values)
            action_advantages = np.vstack(gae_values)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        action_advantages = action_advantages.squeeze(axis=-1)
        actions = batch.actions()
        if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) < 2:
            actions = np.expand_dims(actions, -1)

        # train
        result = self.networks['main'].online_network.accumulate_gradients({**batch.states(network_keys),
                                                                            'output_1_0': actions},
                                                                       [state_value_head_targets, action_advantages])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])

        return total_loss, losses, unclipped_grads

    def get_prediction(self, states):
        tf_input_state = self.prepare_batch_for_inference(states, "main")
        return self.networks['main'].online_network.predict(tf_input_state)[1:]  # index 0 is the state value
