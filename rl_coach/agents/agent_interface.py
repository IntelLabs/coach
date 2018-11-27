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

from typing import Union, List, Dict

import numpy as np

from rl_coach.core_types import EnvResponse, ActionInfo, RunPhase, PredictionType, ActionType, Transition
from rl_coach.saver import SaverCollection


class AgentInterface(object):
    def __init__(self):
        self._phase = RunPhase.HEATUP
        self._parent = None
        self.spaces = None

    @property
    def parent(self):
        """
        Get the parent class of the agent
        :return: the current phase
        """
        return self._parent

    @parent.setter
    def parent(self, val):
        """
        Change the parent class of the agent
        :param val: the new parent
        :return: None
        """
        self._parent = val

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the agent
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the agent
        :param val: the new phase
        :return: None
        """
        self._phase = val

    def reset_internal_state(self) -> None:
        """
        Reset the episode parameters for the agent
        :return: None
        """
        raise NotImplementedError("")

    def train(self) -> Union[float, List]:
        """
        Train the agents network
        :return: The loss of the training
        """
        raise NotImplementedError("")

    def act(self) -> ActionInfo:
        """
        Get a decision of the next action to take.
        The action is dependent on the current state which the agent holds from resetting the environment or from
        the observe function.
        :return: A tuple containing the actual action and additional info on the action
        """
        raise NotImplementedError("")

    def observe(self, env_response: EnvResponse) -> bool:
        """
        Gets a response from the environment.
        Processes this information for later use. For example, create a transition and store it in memory.
        The action info (a class containing any info the agent wants to store regarding its action decision process) is
        stored by the agent itself when deciding on the action.
        :param env_response: a EnvResponse containing the response from the environment
        :return: a done signal which is based on the agent knowledge. This can be different from the done signal from
                 the environment. For example, an agent can decide to finish the episode each time it gets some
                 intrinsic reward
        """
        raise NotImplementedError("")

    def save_checkpoint(self, checkpoint_prefix: str) -> None:
        """
        Save the model of the agent to the disk. This can contain the network parameters, the memory of the agent, etc.
        :param checkpoint_prefix: The prefix of the checkpoint file to save
        :return: None
        """
        raise NotImplementedError("")

    def get_predictions(self, states: Dict, prediction_type: PredictionType) -> np.ndarray:
        """
        Get a prediction from the agent with regard to the requested prediction_type. If the agent cannot predict this
        type of prediction_type, or if there is more than possible way to do so, raise a ValueException.
        :param states:
        :param prediction_type:
        :return: the agent's prediction
        """
        raise NotImplementedError("")

    def set_incoming_directive(self, action: ActionType) -> None:
        """
        Pass a higher level command (directive) to the agent.
        For example, a higher level agent can set the goal of the agent.
        :param action: the directive to pass to the agent
        :return: None
        """
        raise NotImplementedError("")

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #         an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_observe_on_trainer(self, transition: Transition) -> bool:
        """
        This emulates the act using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Gets a response from the environment.
        Processes this information for later use. For example, create a transition and store it in memory.
        The action info (a class containing any info the agent wants to store regarding its action decision process) is
        stored by the agent itself when deciding on the action.
        :param env_response: a EnvResponse containing the response from the environment
        :return: a done signal which is based on the agent knowledge. This can be different from the done signal from
                 the environment. For example, an agent can decide to finish the episode each time it gets some
                 intrinsic reward
        """
        raise NotImplementedError("")

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #         an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_act_on_trainer(self, transition: Transition) -> ActionInfo:
        """
        This emulates the act using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Get a decision of the next action to take.
        The action is dependent on the current state which the agent holds from resetting the environment or from
        the observe function.
        :return: A tuple containing the actual action and additional info on the action
        """
        raise NotImplementedError("")

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collect all of agent savers
        :param parent_path_suffix: path suffix of the parent of the agent
            (could be name of level manager or composite agent)
        :return: collection of all agent savers
        """
        raise NotImplementedError
