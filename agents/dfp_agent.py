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

from agents.agent import *


# Direct Future Prediction Agent - http://vladlen.info/papers/learning-to-act.pdf
class DFPAgent(Agent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        Agent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.current_goal = self.tp.agent.goal_vector
        self.main_network = NetworkWrapper(tuning_parameters, False, self.has_global, 'main',
                                           self.replicated_device, self.worker_device)
        self.networks.append(self.main_network)

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, total_returns = self.extract_batch(batch)

        # create the inputs for the network
        input = current_states
        input['goal'] = np.repeat(np.expand_dims(self.current_goal, 0), self.tp.batch_size, 0)

        # get the current outputs of the network
        targets = self.main_network.online_network.predict(input)

        # change the targets for the taken actions
        for i in range(self.tp.batch_size):
            targets[i, actions[i]] = batch[i].info['future_measurements'].flatten()

        result = self.main_network.train_and_sync_networks(input, targets)
        total_loss = result[0]

        return total_loss

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        # convert to batch so we can run it through the network
        observation = np.expand_dims(np.array(curr_state['observation']), 0)
        measurements = np.expand_dims(np.array(curr_state['measurements']), 0)
        goal = np.expand_dims(self.current_goal, 0)

        # predict the future measurements
        measurements_future_prediction = self.main_network.online_network.predict({
            "observation": observation,
            "measurements": measurements,
            "goal": goal})[0]
        action_values = np.zeros((self.action_space_size,))
        num_steps_used_for_objective = len(self.tp.agent.future_measurements_weights)

        # calculate the score of each action by multiplying it's future measurements with the goal vector
        for action_idx in range(self.action_space_size):
            action_measurements = measurements_future_prediction[action_idx]
            action_measurements = np.reshape(action_measurements,
                                             (self.tp.agent.num_predicted_steps_ahead, self.measurements_size[0]))
            future_steps_values = np.dot(action_measurements, self.current_goal)
            action_values[action_idx] = np.dot(future_steps_values[-num_steps_used_for_objective:],
                                               self.tp.agent.future_measurements_weights)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        if phase == RunPhase.TRAIN:
            action = self.exploration_policy.get_action(action_values)
        else:
            action = np.argmax(action_values)

        action_values = action_values.squeeze()

        # store information for plotting interactively (actual plotting is done in agent)
        if self.tp.visualization.plot_action_values_online:
            for idx, action_name in enumerate(self.env.actions_description):
                self.episode_running_info[action_name].append(action_values[idx])

        action_info = {"action_probability": 0, "action_value": action_values[action]}

        return action, action_info
