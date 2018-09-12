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
from typing import Union

from rl_coach.agents.agent import Agent
from rl_coach.core_types import RunPhase, ActionInfo
from rl_coach.logger import screen
from rl_coach.spaces import DiscreteActionSpace


## This is an abstract agent - there is no learn_from_batch method ##

# Imitation Agent
class ImitationAgent(Agent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.imitation = True

    def extract_action_values(self, prediction):
        return prediction.squeeze()

    def choose_action(self, curr_state):
        # convert to batch so we can run it through the network
        prediction = self.networks['main'].online_network.predict(self.prepare_batch_for_inference(curr_state, 'main'))

        # get action values and extract the best action from it
        action_values = self.extract_action_values(prediction)
        self.exploration_policy.change_phase(RunPhase.TEST)
        action = self.exploration_policy.get_action(action_values)
        action_info = ActionInfo(action=action)

        return action_info

    def log_to_screen(self):
        # log to screen
        if self.phase == RunPhase.TRAIN:
            # for the training phase - we log during the episode to visualize the progress in training
            log = OrderedDict()
            if self.task_id is not None:
                log["Worker"] = self.task_id
            log["Episode"] = self.current_episode
            log["Loss"] = self.loss.values[-1]
            log["Training iteration"] = self.training_iteration
            screen.log_dict(log, prefix="Training")
        else:
            # for the evaluation phase - logging as in regular RL
            super().log_to_screen()

    def learn_from_batch(self, batch):
        raise NotImplementedError("ImitationAgent is an abstract agent. Not to be used directly.")
