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

import copy
import itertools
from enum import Enum
from typing import Union, List, Dict

import numpy as np

from rl_coach.agents.agent_interface import AgentInterface
from rl_coach.base_parameters import AgentParameters, VisualizationParameters
from rl_coach.core_types import ActionInfo, EnvResponse, ActionType, RunPhase
from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.saver import SaverCollection
from rl_coach.spaces import ActionSpace
from rl_coach.spaces import AgentSelection, AttentionActionSpace, SpacesDefinition
from rl_coach.utils import short_dynamic_import


class DecisionPolicy(object):
    def choose_action(self, actions_info: Dict[str, ActionInfo]) -> ActionInfo:
        """
        Given a list of actions from multiple agents, decide on a single action to take.
        :param actions_info: a dictionary of agent names and their corresponding
                             ActionInfo instances containing information for each agents action
        :return: a single action and the corresponding action info
        """
        raise NotImplementedError("")


class SingleDecider(DecisionPolicy):
    """
    A decision policy that chooses the action according to the agent that is currently in control.
    """
    def __init__(self, default_decision_maker: str):
        super().__init__()
        self._decision_maker = default_decision_maker

    @property
    def decision_maker(self):
        """
        Get the decision maker that was set by the upper level control.
        """
        return self._decision_maker

    @decision_maker.setter
    def decision_maker(self, decision_maker: str):
        """
        Set the decision maker by the upper level control.
        :param action: the incoming action from the upper level control.
        """
        self._decision_maker = decision_maker

    def choose_action(self, actions_info: Dict[str, ActionInfo]) -> ActionInfo:
        """
        Given a list of actions from multiple agents, take the action of the current decision maker
        :param actions_info: a list of ActionInfo instances containing the information for each agents action
        :return: a single action
        """
        if self.decision_maker not in actions_info.keys():
            raise ValueError("The current decision maker ({}) does not exist in the given actions ({})"
                             .format(self.decision_maker, actions_info.keys()))
        return actions_info[self.decision_maker]


class RoundRobin(DecisionPolicy):
    """
    A decision policy that chooses the action according to agents selected in a circular order.
    """
    def __init__(self, num_agents: int):
        super().__init__()
        self.round_robin = itertools.cycle(range(num_agents))

    def choose_action(self, actions_info: Dict[str, ActionInfo]) -> ActionInfo:
        """
        Given a list of actions from multiple agents, take the action of the current decision maker, which is set in a
         circular order
        :param actions_info: a list of ActionInfo instances containing the information for each agents action
        :return: a single action
        """
        decision_maker = self.round_robin.__next__()
        if decision_maker not in range(len(actions_info.keys())):
            raise ValueError("The size of action_info does not match the number of agents set to RoundRobin decision"
                             " policy.")
        return actions_info.items()[decision_maker]


class MajorityVote(DecisionPolicy):
    """
    A decision policy that chooses the action that most of the agents chose.
    This policy is only useful for discrete control.
    """
    def __init__(self):
        super().__init__()

    def choose_action(self, actions_info: Dict[str, ActionInfo]) -> ActionInfo:
        """
        Given a list of actions from multiple agents, take the action that most agents agree on
        :param actions_info: a list of ActionInfo instances containing the information for each agents action
        :return: a single action
        """
        # TODO: enforce discrete action spaces
        if len(actions_info.keys()) == 0:
            raise ValueError("The given list of actions is empty")
        vote_count = np.bincount([action_info.action for action_info in actions_info.values()])
        majority_vote = np.argmax(vote_count)
        return actions_info.items()[majority_vote]


class MeanDecision(DecisionPolicy):
    """
    A decision policy that takes the mean action given the actions of all the agents.
    This policy is only useful for continuous control.
    """
    def __init__(self):
        super().__init__()

    def choose_action(self, actions_info: Dict[str, ActionInfo]) -> ActionInfo:
        """
        Given a list of actions from multiple agents, take the mean action
        :param actions_info: a list of ActionInfo instances containing the information for each agents action
        :return: a single action
        """
        # TODO: enforce continuous action spaces
        if len(actions_info.keys()) == 0:
            raise ValueError("The given list of actions is empty")
        mean = np.mean([action_info.action for action_info in actions_info.values()], axis=0)
        return ActionInfo(mean)


class RewardPolicy(Enum):
    ReachingGoal = 0
    NativeEnvironmentReward = 1
    AccumulatedEnvironmentRewards = 2


class CompositeAgent(AgentInterface):
    """
    A CompositeAgent is a group of agents in the same hierarchy level.
    In a CompositeAgent, each agent may take the role of either a controller or an observer.
    Each agent that is defined as observer, gets observations from the environment.
    Each agent that is defined as controller, can potentially also control the environment, in addition to observing it.
    There are several ways to decide on the action from different controller agents:
    1. Ensemble -
        - Take the majority vote (discrete controls)
        - Take the mean action (continuous controls)
        - Round robin between the agents (discrete/continuous)
    2. Skills -
        - At each step a single agent decides (Chosen by the uppoer hierarchy controlling agent)

    A CompositeAgent can be controlled using one of the following methods (ActionSpaces):
    1. Goals (in terms of measurements, observation, embedding or a change in those values)
    2. Agent Selection (skills) / Discrete action space.
    3. Attention (a subset of the real environment observation / action space)
    """
    def __init__(self,
                 agents_parameters: Union[AgentParameters, Dict[str, AgentParameters]],
                 visualization_parameters: VisualizationParameters,
                 decision_policy: DecisionPolicy,
                 out_action_space: ActionSpace,
                 in_action_space: Union[None, ActionSpace]=None,
                 decision_makers: Union[bool, Dict[str, bool]]=True,
                 reward_policy: RewardPolicy=RewardPolicy.NativeEnvironmentReward,
                 name="CompositeAgent"):
        """
        Construct an agent group
        :param agents_parameters: a list of presets describing each one of the agents in the group
        :param decision_policy: the decision policy of the group which describes how actions are consolidated
        :param out_action_space: the type of action space that is used by this composite agent in order to control the
                                 underlying environment
        :param in_action_space: the type of action space that is used by the upper level agent in order to control this
                                group
        :param decision_makers: a list of booleans representing for each corresponding agent if it has a decision
                                privilege or if it is just an observer
        :param reward_policy: the type of the reward that the group receives
        """
        super().__init__()

        if isinstance(agents_parameters, AgentParameters):
            decision_makers = {agents_parameters.name: True}
            agents_parameters = {agents_parameters.name: agents_parameters}
        self.agents_parameters = agents_parameters
        self.visualization_parameters = visualization_parameters
        self.decision_makers = decision_makers
        self.decision_policy = decision_policy
        self.in_action_space = in_action_space
        self.out_action_space = out_action_space  # TODO: this is not being used
        self.reward_policy = reward_policy
        self.full_name_id = self.name = name
        self.current_decision_maker = 0
        self.environment = None
        self.agents = {}  # key = agent_name, value = agent
        self.incoming_action = None
        self.last_state = None
        self._phase = RunPhase.HEATUP
        self.last_action_info = None
        self.current_episode = 0
        self.parent_level_manager = None

        # environment spaces
        self.spaces = None

        # counters for logging
        self.total_steps_counter = 0
        self.current_episode_steps_counter = 0
        self.total_reward_in_current_episode = 0

        # validate input
        if set(self.decision_makers) != set(self.agents_parameters):
            raise ValueError("The decision_makers dictionary keys does not match the names of the given agents")
        if sum(self.decision_makers.values()) > 1 and type(self.decision_policy) == SingleDecider \
                and type(self.in_action_space) != AgentSelection:
            raise ValueError("When the control policy is set to single decider, the master policy should control the"
                             "agent group via agent selection (ControlType.AgentSelection)")

    @property
    def parent(self):
        """
        Get the parent class of the composite agent
        :return: the current phase
        """
        return self._parent

    @parent.setter
    def parent(self, val):
        """
        Change the parent class of the composite agent.
        Additionally, updates the full name of the agent
        :param val: the new parent
        :return: None
        """
        self._parent = val
        if not hasattr(self._parent, 'name'):
            raise ValueError("The parent of a composite agent must have a name")
        self.full_name_id = "{}/{}".format(self._parent.name, self.name)

    def create_agents(self):
        for agent_name, agent_parameters in self.agents_parameters.items():
            agent_parameters.name = agent_name

            # create agent
            self.agents[agent_parameters.name] = short_dynamic_import(agent_parameters.path)(agent_parameters,
                                                                                             parent=self)
            self.agents[agent_parameters.name].parent_level_manager = self.parent_level_manager

        # TODO: this is a bit too specific to be defined here
        # add an attention cropping filter if the incoming directives are attention boxes
        if isinstance(self.in_action_space, AttentionActionSpace):
            attention_size = self.in_action_space.forced_attention_size
            for agent in self.agents.values():
                agent.input_filter.observation_filters['attention'] = \
                    ObservationCropFilter(crop_low=np.zeros_like(attention_size), crop_high=attention_size)
                agent.input_filter.observation_filters.move_to_end('attention', last=False)  # add the cropping at the beginning

    def setup_logger(self) -> None:
        """
        Setup the logger for all the agents in the composite agent
        :return: None
        """
        [agent.setup_logger() for agent in self.agents.values()]

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the agents in the composite agent
        :return: None
        """
        [agent.set_session(sess) for agent in self.agents.values()]

    def set_environment_parameters(self, spaces: SpacesDefinition):
        """
        Sets the parameters that are environment dependent. As a side effect, initializes all the components that are
        dependent on those values, by calling init_environment_dependent_modules
        :param spaces: the definitions of all the spaces of the environment
        :return: None
        """
        self.spaces = copy.deepcopy(spaces)
        [agent.set_environment_parameters(self.spaces) for agent in self.agents.values()]

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase) -> None:
        """
        Change the current phase of all the agents in the group
        :param phase: the new phase
        :return: None
        """
        self._phase = val
        for agent in self.agents.values():
            agent.phase = val

    def end_episode(self) -> None:
        """
        End an episode
        :return: None
        """
        self.current_episode += 1
        [agent.handle_episode_ended() for agent in self.agents.values()]

    def reset_internal_state(self) -> None:
        """
        Reset the episode for all the agents in the group
        :return: None
        """
        # update counters
        self.total_steps_counter = 0
        self.current_episode_steps_counter = 0
        self.total_reward_in_current_episode = 0

        # reset all sub modules
        [agent.reset_internal_state() for agent in self.agents.values()]

    def train(self) -> Union[float, List]:
        """
        Make a single training step for all the agents of the group
        :return: a list of loss values from the training step
        """
        return [agent.train() for agent in self.agents.values()]

    def act(self) -> ActionInfo:
        """
        Get the actions from all the agents in the group. Then use the decision policy in order to
        extract a single action out of the list of actions.
        :return: the chosen action and its corresponding information
        """

        # update counters
        self.total_steps_counter += 1
        self.current_episode_steps_counter += 1

        # get the actions info from all the agents
        actions_info = {}
        for agent_name, agent in self.agents.items():
            action_info = agent.act()
            actions_info[agent_name] = action_info

        # decide on a single action to apply to the environment
        action_info = self.decision_policy.choose_action(actions_info)

        # TODO: make the last action info a property?
        # pass the action info to all the observers
        for agent_name, is_decision_maker in self.decision_makers.items():
            if not is_decision_maker:
                self.agents[agent_name].last_action_info = action_info
        self.last_action_info = action_info

        return self.last_action_info

    def observe(self, env_response: EnvResponse) -> bool:
        """
        Given a response from the environment as a env_response, filter it and pass it to the agents.
        This method has two main jobs:
        1. Wrap the previous transition, ending with the new observation coming from EnvResponse.
        2. Save the next_state as the current_state to take action upon for the next call to act().

        :param env_response:
        :param action_info: additional info about the chosen action
        :return:
        """

        # accumulate the unfiltered rewards for visualization
        self.total_reward_in_current_episode += env_response.reward

        episode_ended = env_response.game_over

        # pass the env_response to all the sub-agents
        # TODO: what if one agent decides to end the episode but the others don't? who decides?
        for agent_name, agent in self.agents.items():
            goal_reached = agent.observe(env_response)
            episode_ended = episode_ended or goal_reached

        # TODO: unlike for a single agent, here we also treat a game over by the environment.
        # probably better to only return the agents' goal_reached decisions.
        return episode_ended

    def save_checkpoint(self, checkpoint_prefix: str) -> None:
        [agent.save_checkpoint(checkpoint_prefix) for agent in self.agents.values()]

    def restore_checkpoint(self, checkpoint_dir: str) -> None:
        [agent.restore_checkpoint(checkpoint_dir) for agent in self.agents.values()]

    def set_incoming_directive(self, action: ActionType) -> None:
        self.incoming_action = action
        if isinstance(self.decision_policy, SingleDecider) and isinstance(self.in_action_space, AgentSelection):
            self.decision_policy.decision_maker = list(self.agents.keys())[action]
        if isinstance(self.in_action_space, AttentionActionSpace):
            # TODO: redesign to be more modular
            for agent in self.agents.values():
                agent.input_filter.observation_filters['attention'].crop_low = action[0]
                agent.input_filter.observation_filters['attention'].crop_high = action[1]
                agent.output_filter.action_filters['masking'].set_masking(action[0], action[1])

        # TODO  rethink this scheme. we don't want so many if else clauses lying around here.  
        # TODO - for incoming actions which do not involve setting the acting agent we should change the
        #  observation_space, goal to pursue, etc accordingly to the incoming action.

    def sync(self) -> None:
        """
        Sync the agent networks with the global network
        :return:
        """
        [agent.sync() for agent in self.agents.values()]

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collect all of agent's network savers
        :param parent_path_suffix: path suffix of the parent of the agent
            (could be name of level manager or composite agent)
        :return: collection of all agent savers
        """
        savers = SaverCollection()
        for agent in self.agents.values():
            savers.update(agent.collect_savers(
                parent_path_suffix="{}.{}".format(parent_path_suffix, self.name)))
        return savers
