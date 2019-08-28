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

from typing import Union
import copy
import numpy as np
from collections import OrderedDict

from rl_coach.agents.agent import Agent
from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent

from rl_coach.architectures.head_parameters import SACQHeadParameters,SACPolicyHeadParameters,VHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, AgentParameters, EmbedderScheme, MiddlewareScheme
from rl_coach.core_types import ActionInfo, EnvironmentSteps, RunPhase
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.spaces import BoxActionSpace


# There are 3 networks in SAC implementation. All have the same topology but parameters are not shared.
# The networks are:
# 1. State Value Network - SACValueNetwork
# 2. Soft Q Value Network - SACCriticNetwork
# 3. Policy Network - SACPolicyNetwork - currently supporting only Gaussian Policy


# 1. State Value Network - SACValueNetwork
# this is the state value network in SAC.
# The network is trained to predict (regression) the state value in the max-entropy settings
# The objective to be minimized is given in equation (5) in the paper:
#
# J(psi)= E_(s~D)[0.5*(V_psi(s)-y(s))^2]
# where y(s) = E_(a~pi)[Q_theta(s,a)-log(pi(a|s))]


# Default parameters for value network:
# topology :
#   input embedder : EmbedderScheme.Medium (Dense(256)) , relu activation
#   middleware : EmbedderScheme.Medium (Dense(256)) , relu activation


class SACValueNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='relu')}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='relu')
        self.heads_parameters = [VHeadParameters(initializer='xavier')]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003     # 3e-4 see appendix D in the paper
        self.create_target_network = True   # tau is set in SoftActorCriticAlgorithmParameters.rate_for_copying_weights_to_target


# 2. Soft Q Value Network - SACCriticNetwork
# the whole network is built in the SACQHeadParameters. we use empty input embedder and middleware
class SACCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Empty)
        self.heads_parameters = [SACQHeadParameters()]      # SACQHeadParameters includes the topology of the head
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = False


# 3. policy Network
# Default parameters for policy network:
# topology :
#   input embedder : EmbedderScheme.Medium (Dense(256)) , relu activation
#   middleware : EmbedderScheme = [Dense(256)] , relu activation --> scheme should be overridden in preset
class SACPolicyNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(activation_function='relu')}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='relu')
        self.heads_parameters = [SACPolicyHeadParameters()]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = False
        self.l2_regularization = 0      # weight decay regularization. not used in the original paper


# Algorithm Parameters

class SoftActorCriticAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_copying_online_weights_to_target: (StepMethod)
        The number of steps between copying the online network weights to the target network weights.

    :param rate_for_copying_weights_to_target: (float)
        When copying the online network weights to the target network weights, a soft update will be used, which
        weight the new online network weights by rate_for_copying_weights_to_target. (Tau as defined in the paper)

    :param use_deterministic_for_evaluation: (bool)
        If True, during the evaluation phase, action are chosen deterministically according to the policy mean
        and not sampled from the policy distribution.
    """
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.005
        self.use_deterministic_for_evaluation = True    # evaluate agent using deterministic policy (i.e. take the mean value)


class SoftActorCriticAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=SoftActorCriticAlgorithmParameters(),
                         exploration=AdditiveNoiseParameters(),
                         memory=ExperienceReplayParameters(),   # SAC doesnt use episodic related data
                         # network wrappers:
                         networks=OrderedDict([("policy", SACPolicyNetworkParameters()),
                                               ("q", SACCriticNetworkParameters()),
                                               ("v", SACValueNetworkParameters())]))

    @property
    def path(self):
        return 'rl_coach.agents.soft_actor_critic_agent:SoftActorCriticAgent'


# Soft Actor Critic - https://arxiv.org/abs/1801.01290
class SoftActorCriticAgent(PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.last_gradient_update_step_idx = 0

        # register signals to track (in learn_from_batch)
        self.policy_means = self.register_signal('Policy_mu_avg')
        self.policy_logsig = self.register_signal('Policy_logsig')
        self.policy_logprob_sampled = self.register_signal('Policy_logp_sampled')
        self.policy_grads = self.register_signal('Policy_grads_sumabs')

        self.q1_values = self.register_signal("Q1")
        self.TD_err1 = self.register_signal("TD err1")
        self.q2_values = self.register_signal("Q2")
        self.TD_err2 = self.register_signal("TD err2")
        self.v_tgt_ns = self.register_signal('V_tgt_ns')
        self.v_onl_ys = self.register_signal('V_onl_ys')
        self.action_signal = self.register_signal("actions")

    def learn_from_batch(self, batch):
        #########################################
        # need to update the following networks:
        # 1. actor (policy)
        # 2. state value (v)
        # 3. critic (q1 and q2)
        # 4. target network - probably already handled by V

        #########################################
        # define the networks to be used

        # State Value Network
        value_network = self.networks['v']
        value_network_keys = self.ap.network_wrappers['v'].input_embedders_parameters.keys()

        # Critic Network
        q_network = self.networks['q'].online_network
        q_head = q_network.output_heads[0]
        q_network_keys = self.ap.network_wrappers['q'].input_embedders_parameters.keys()

        # Actor (policy) Network
        policy_network = self.networks['policy'].online_network
        policy_network_keys = self.ap.network_wrappers['policy'].input_embedders_parameters.keys()

        ##########################################
        # 1. updating the actor - according to (13) in the paper
        policy_inputs = copy.copy(batch.states(policy_network_keys))
        policy_results = policy_network.predict(policy_inputs)

        policy_mu, policy_std, sampled_raw_actions, sampled_actions, sampled_actions_logprob, \
        sampled_actions_logprob_mean = policy_results

        self.policy_means.add_sample(policy_mu)
        self.policy_logsig.add_sample(policy_std)
        self.policy_logprob_sampled.add_sample(sampled_actions_logprob_mean)

        # get the state-action values for the replayed states and their corresponding actions from the policy
        q_inputs = copy.copy(batch.states(q_network_keys))
        q_inputs['output_0_0'] = sampled_actions
        log_target = q_network.predict(q_inputs)[0].squeeze()

        # log internal q values
        q1_vals, q2_vals = q_network.predict(q_inputs, outputs=[q_head.q1_output, q_head.q2_output])
        self.q1_values.add_sample(q1_vals)
        self.q2_values.add_sample(q2_vals)

        # calculate the gradients according to (13)
        # get the gradients of log_prob w.r.t the weights (parameters) - indicated as phi in the paper
        initial_feed_dict = {policy_network.gradients_weights_ph[5]: np.array(1.0)}
        dlogp_dphi = policy_network.predict(policy_inputs,
                                            outputs=policy_network.weighted_gradients[5],
                                            initial_feed_dict=initial_feed_dict)

        # calculate dq_da
        dq_da = q_network.predict(q_inputs,
                                  outputs=q_network.gradients_wrt_inputs[1]['output_0_0'])

        # calculate da_dphi
        initial_feed_dict = {policy_network.gradients_weights_ph[3]: dq_da}
        dq_dphi = policy_network.predict(policy_inputs,
                                         outputs=policy_network.weighted_gradients[3],
                                         initial_feed_dict=initial_feed_dict)

        # now given dlogp_dphi, dq_dphi we need to calculate the policy gradients according to (13)
        policy_grads = [dlogp_dphi[l] - dq_dphi[l] for l in range(len(dlogp_dphi))]

        # apply the gradients to policy networks
        policy_network.apply_gradients(policy_grads)
        grads_sumabs = np.sum([np.sum(np.abs(policy_grads[l])) for l in range(len(policy_grads))])
        self.policy_grads.add_sample(grads_sumabs)

        ##########################################
        # 2. updating the state value online network weights
        # done by calculating the targets for the v head according to (5) in the paper
        # value_targets = log_targets-sampled_actions_logprob
        value_inputs = copy.copy(batch.states(value_network_keys))
        value_targets = log_target - sampled_actions_logprob

        self.v_onl_ys.add_sample(value_targets)

        # call value_network apply gradients with this target
        value_loss = value_network.online_network.train_on_batch(value_inputs, value_targets[:,None])[0]

        ##########################################
        # 3. updating the critic (q networks)
        # updating q networks according to (7) in the paper

        # define the input to the q network: state has been already updated previously, but now we need
        # the actions from the batch (and not those sampled by the policy)
        q_inputs['output_0_0'] = batch.actions(len(batch.actions().shape) == 1)

        # define the targets : scale_reward * reward + (1-terminal)*discount*v_target_next_state
        # define v_target_next_state
        value_inputs = copy.copy(batch.next_states(value_network_keys))
        v_target_next_state = value_network.target_network.predict(value_inputs)
        self.v_tgt_ns.add_sample(v_target_next_state)
        # Note: reward is assumed to be rescaled by RewardRescaleFilter in the preset parameters
        TD_targets = batch.rewards(expand_dims=True) + \
                     (1.0 - batch.game_overs(expand_dims=True)) * self.ap.algorithm.discount * v_target_next_state

        # call critic network update
        result = q_network.train_on_batch(q_inputs, TD_targets, additional_fetches=[q_head.q1_loss, q_head.q2_loss])
        total_loss, losses, unclipped_grads = result[:3]
        q1_loss, q2_loss = result[3]
        self.TD_err1.add_sample(q1_loss)
        self.TD_err2.add_sample(q2_loss)

        ##########################################
        # 4. updating the value target network
        # I just need to set the parameter rate_for_copying_weights_to_target in the agent parameters to be 1-tau
        # where tau is the hyper parameter as defined in sac original implementation

        return total_loss, losses, unclipped_grads

    def get_prediction(self, states):
        """
        get the mean and stdev of the policy distribution given 'states'
        :param states: the states for which we need to sample actions from the policy
        :return: mean and stdev
        """
        tf_input_state = self.prepare_batch_for_inference(states, 'policy')
        return self.networks['policy'].online_network.predict(tf_input_state)

    def train(self):
        # since the algorithm works with experience replay buffer (non-episodic),
        # we cant use the policy optimization train method. we need Agent.train
        # note that since in Agent.train there is no apply_gradients, we need to do it in learn from batch
        return Agent.train(self)

    def choose_action(self, curr_state):
        """
        choose_action - chooses the most likely action
        if 'deterministic' - take the mean of the policy which is the prediction of the policy network.
        else - use the exploration policy
        :param curr_state:
        :return: action wrapped in ActionInfo
        """
        if not isinstance(self.spaces.action, BoxActionSpace):
            raise ValueError("SAC works only for continuous control problems")
        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'policy')
        # use the online network for prediction
        policy_network = self.networks['policy'].online_network
        policy_head = policy_network.output_heads[0]
        result = policy_network.predict(tf_input_state,
                                        outputs=[policy_head.policy_mean, policy_head.actions])
        action_mean, action_sample = result

        # if using deterministic policy, take the mean values. else, use exploration policy to sample from the pdf
        if self.phase == RunPhase.TEST and self.ap.algorithm.use_deterministic_for_evaluation:
            action = action_mean[0]
        else:
            action = action_sample[0]

        self.action_signal.add_sample(action)

        action_info = ActionInfo(action=action)
        return action_info
