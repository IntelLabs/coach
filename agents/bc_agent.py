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

import numpy as np

from agents.imitation_agent import ImitationAgent


# Behavioral Cloning Agent
class BCAgent(ImitationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ImitationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)

    def learn_from_batch(self, batch):
        current_states, _, actions, _, _, _ = self.extract_batch(batch)

        # the targets for the network are the actions since this is supervised learning
        if self.env.discrete_controls:
            targets = np.eye(self.env.action_space_size)[[actions]]
        else:
            targets = actions

        result = self.main_network.train_and_sync_networks(current_states, targets)
        total_loss = result[0]

        return total_loss
