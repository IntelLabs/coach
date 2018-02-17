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

from agents.actor_critic_agent import *
from configurations import *


# Deep Deterministic Policy Gradients Network - https://arxiv.org/pdf/1509.02971.pdf
class DDPGAgent(ActorCriticAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ActorCriticAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id,
                                  create_target_network=True)
        # define critic network
        self.critic_network = self.main_network
        # self.networks.append(self.critic_network)

        # define actor network
        tuning_parameters.agent.input_types = {'observation': InputTypes.Observation}
        tuning_parameters.agent.output_types = [OutputTypes.Pi]
        self.actor_network = NetworkWrapper(tuning_parameters, True, self.has_global, 'actor',
                                            self.replicated_device, self.worker_device)
        self.networks.append(self.actor_network)

        self.q_values = Signal("Q")
        self.signals.append(self.q_values)

        self.reset_game(do_not_reset_env=True)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)

        # TD error = r + discount*max(q_st_plus_1) - q_st
        next_actions = self.actor_network.target_network.predict(next_states)
        inputs = copy.copy(next_states)
        inputs['action'] = next_actions
        q_st_plus_1 = self.critic_network.target_network.predict(inputs)
        TD_targets = np.expand_dims(rewards, -1) + \
                     (1.0 - np.expand_dims(game_overs, -1)) * self.tp.agent.discount * q_st_plus_1

        # get the gradients of the critic output with respect to the action
        actions_mean = self.actor_network.online_network.predict(current_states)
        critic_online_network = self.critic_network.online_network
        # TODO: convert into call to predict, current method ignores lstm middleware for example
        action_gradients = self.critic_network.sess.run(critic_online_network.gradients_wrt_inputs['action'],
                                                        feed_dict=critic_online_network._feed_dict({
                                                            **current_states,
                                                            'action': actions_mean,
                                                        }))[0]

        # train the critic
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, -1)
        result = self.critic_network.train_and_sync_networks({**current_states, 'action': actions}, TD_targets)
        total_loss = result[0]

        # apply the gradients from the critic to the actor
        actor_online_network = self.actor_network.online_network
        gradients = self.actor_network.sess.run(actor_online_network.weighted_gradients,
                                                feed_dict=actor_online_network._feed_dict({
                                                    **current_states,
                                                    actor_online_network.gradients_weights_ph: -action_gradients,
                                                }))
        if self.actor_network.has_global:
            self.actor_network.global_network.apply_gradients(gradients)
            self.actor_network.update_online_network()
        else:
            self.actor_network.online_network.apply_gradients(gradients)

        return total_loss

    def train(self):
        return Agent.train(self)

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        assert not self.env.discrete_controls, 'DDPG works only for continuous control problems'
        result = self.actor_network.online_network.predict(self.tf_input_state(curr_state))
        action_values = result[0].squeeze()

        if phase == RunPhase.TRAIN:
            action = self.exploration_policy.get_action(action_values)
        else:
            action = action_values

        action = np.clip(action, self.env.action_space_low, self.env.action_space_high)

        # get q value
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        inputs = self.tf_input_state(curr_state)
        inputs['action'] = action_batch
        q_value = self.critic_network.online_network.predict(inputs)[0]
        self.q_values.add_sample(q_value)
        action_info = {"action_value": q_value}

        return action, action_info
