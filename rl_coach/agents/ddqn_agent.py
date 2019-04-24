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
from rl_coach.agents.dqn_agent import DQNAgent, DQNAgentParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.schedules import LinearSchedule


class DDQNAgentParameters(DQNAgentParameters):
    def __init__(self):
        super().__init__()
        self.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(30000)
        self.exploration.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
        self.exploration.evaluation_epsilon = 0.001

    @property
    def path(self):
        return 'rl_coach.agents.ddqn_agent:DDQNAgent'


# Double DQN - https://arxiv.org/abs/1509.06461
class DDQNAgent(DQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def select_actions(self, next_states, q_st_plus_1):
        return np.argmax(self.networks['main'].online_network.predict(next_states), 1)

