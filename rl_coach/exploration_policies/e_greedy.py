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

from typing import List

import numpy as np

from rl_coach.core_types import RunPhase, ActionType
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.exploration_policy import ExplorationParameters, ExplorationPolicy
from rl_coach.schedules import Schedule, LinearSchedule
from rl_coach.spaces import ActionSpace, DiscreteActionSpace, BoxActionSpace
from rl_coach.utils import dynamic_import_and_instantiate_module_from_params


class EGreedyParameters(ExplorationParameters):
    def __init__(self):
        super().__init__()
        self.epsilon_schedule = LinearSchedule(0.5, 0.01, 50000)
        self.evaluation_epsilon = 0.05
        self.continuous_exploration_policy_parameters = AdditiveNoiseParameters()
        self.continuous_exploration_policy_parameters.noise_schedule = LinearSchedule(0.1, 0.1, 50000)
        # for continuous control -
        # (see http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)

    @property
    def path(self):
        return 'rl_coach.exploration_policies.e_greedy:EGreedy'


class EGreedy(ExplorationPolicy):
    """
    e-greedy is an exploration policy that is intended for both discrete and continuous action spaces.

    For discrete action spaces, it assumes that each action is assigned a value, and it selects the action with the
    highest value with probability 1 - epsilon. Otherwise, it selects a action sampled uniformly out of all the
    possible actions. The epsilon value is given by the user and can be given as a schedule.
    In evaluation, a different epsilon value can be specified.

    For continuous action spaces, it assumes that the mean action is given by the agent. With probability epsilon,
    it samples a random action out of the action space bounds. Otherwise, it selects the action according to a
    given continuous exploration policy, which is set to AdditiveNoise by default. In evaluation, the action is
    always selected according to the given continuous exploration policy (where its phase is set to evaluation as well).
    """
    def __init__(self, action_space: ActionSpace, epsilon_schedule: Schedule,
                 evaluation_epsilon: float,
                 continuous_exploration_policy_parameters: ExplorationParameters=AdditiveNoiseParameters()):
        """
        :param action_space: the action space used by the environment
        :param epsilon_schedule: a schedule for the epsilon values
        :param evaluation_epsilon: the epsilon value to use for evaluation phases
        :param continuous_exploration_policy_parameters: the parameters of the continuous exploration policy to use
                                                         if the e-greedy is used for a continuous policy
        """
        super().__init__(action_space)
        self.epsilon_schedule = epsilon_schedule
        self.evaluation_epsilon = evaluation_epsilon

        if isinstance(self.action_space, BoxActionSpace):
            # for continuous e-greedy (see http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)
            continuous_exploration_policy_parameters.action_space = action_space
            self.continuous_exploration_policy = \
                dynamic_import_and_instantiate_module_from_params(continuous_exploration_policy_parameters)

        self.current_random_value = np.random.rand()

    def requires_action_values(self):
        epsilon = self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon_schedule.current_value
        return self.current_random_value >= epsilon

    def get_action(self, action_values: List[ActionType]) -> (ActionType, List[float]):
        epsilon = self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon_schedule.current_value

        if isinstance(self.action_space, DiscreteActionSpace):
            if self.current_random_value < epsilon:
                chosen_action = self.action_space.sample()
                probabilities = np.full(len(self.action_space.actions),
                                      1. / (self.action_space.high[0] - self.action_space.low[0] + 1))
            else:
                chosen_action = np.argmax(np.random.random(action_values.shape) *
                                          (np.isclose(action_values, action_values.max())))

                # one-hot probabilities vector
                probabilities = np.zeros(len(self.action_space.actions))
                probabilities[chosen_action] = 1

            self.step_epsilon()
            return chosen_action, probabilities

        else:
            if self.current_random_value < epsilon and self.phase == RunPhase.TRAIN:
                chosen_action = self.action_space.sample()
            else:
                chosen_action = self.continuous_exploration_policy.get_action(action_values)

            self.step_epsilon()
            return chosen_action

    def get_control_param(self):
        if isinstance(self.action_space, DiscreteActionSpace):
            return self.evaluation_epsilon if self.phase == RunPhase.TEST else self.epsilon_schedule.current_value
        elif isinstance(self.action_space, BoxActionSpace):
            return self.continuous_exploration_policy.get_control_param()

    def change_phase(self, phase):
        super().change_phase(phase)
        if isinstance(self.action_space, BoxActionSpace):
            self.continuous_exploration_policy.change_phase(phase)

    def step_epsilon(self):
        # step the epsilon schedule and generate a new random value for next time
        if self.phase == RunPhase.TRAIN:
            self.epsilon_schedule.step()
        self.current_random_value = np.random.rand()
