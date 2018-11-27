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
from typing import Union, Dict

from rl_coach.agents.composite_agent import CompositeAgent
from rl_coach.core_types import EnvResponse, ActionInfo, RunPhase, ActionType, EnvironmentSteps, Transition
from rl_coach.environments.environment import Environment
from rl_coach.environments.environment_interface import EnvironmentInterface
from rl_coach.saver import SaverCollection
from rl_coach.spaces import ActionSpace, SpacesDefinition


class LevelManager(EnvironmentInterface):
    """
    The LevelManager is in charge of managing a level in the hierarchy of control. Each level can have one or more
    CompositeAgents and an environment to control. Its API is double-folded:
        1. Expose services of a LevelManager such as training the level, or stepping it (while behaving according to a
           LevelBehaviorScheme, e.g. as SelfPlay between two identical agents). These methods are implemented in the
           LevelManagerLogic class.
        2. Disguise as appearing as an environment to the upper level control so it will believe it is interacting with
           an environment. This includes stepping through what appears to be a regular environment, setting its phase
           or resetting it. These methods are implemented directly in LevelManager as it inherits from
           EnvironmentInterface.
    """
    def __init__(self,
                 name: str,
                 agents: Union['Agent', CompositeAgent, Dict[str, Union['Agent', CompositeAgent]]],
                 environment: Union['LevelManager', Environment],
                 real_environment: Environment=None,
                 steps_limit: EnvironmentSteps=EnvironmentSteps(1),
                 should_reset_agent_state_after_time_limit_passes: bool=False
                 ):
        """
        A level manager controls a single or multiple composite agents and a single environment.
        The environment can be either a real environment or another level manager behaving as an environment.
        :param agents: a list of agents or composite agents to control
        :param environment: an environment or level manager to control
        :param real_environment: the real environment that is is acted upon. if this is None (which it should be for
         the most bottom level), it will be replaced by the environment parameter. For simple RL schemes, where there
         is only a single level of hierarchy, this removes the requirement of defining both the environment and the
         real environment, as they are the same.
        :param steps_limit: the number of time steps to run when stepping the internal components
        :param should_reset_agent_state_after_time_limit_passes: reset the agent after stepping for steps_limit
        :param name: the level's name
        """
        super().__init__()

        if not isinstance(agents, dict):
            # insert the single composite agent to a dictionary for compatibility
            agents = {agents.name: agents}
        if real_environment is None:
            self._real_environment = real_environment = environment
        self.agents = agents
        self.environment = environment
        self.real_environment = real_environment
        self.steps_limit = steps_limit
        self.should_reset_agent_state_after_time_limit_passes = should_reset_agent_state_after_time_limit_passes
        self.full_name_id = self.name = name
        self._phase = RunPhase.HEATUP
        self.reset_required = False

        # set self as the parent for all the composite agents
        for agent in self.agents.values():
            agent.parent = self
            agent.parent_level_manager = self

        # create all agents in all composite_agents - we do it here so agents will have access to their level manager
        for agent in self.agents.values():
            if isinstance(agent, CompositeAgent):
                agent.create_agents()

        if not isinstance(self.steps_limit, EnvironmentSteps):
            raise ValueError("The num consecutive steps for acting must be defined in terms of environment steps")
        self.build()

        self.last_env_response = self.real_environment.last_env_response
        self.parent_graph_manager = None

    def handle_episode_ended(self) -> None:
        """
        End the environment episode
        :return: None
        """
        [agent.handle_episode_ended() for agent in self.agents.values()]

    def reset_internal_state(self, force_environment_reset: bool = False) -> EnvResponse:
        """
        Reset the environment episode parameters
        :param force_enviro nment_reset: in some cases, resetting the environment can be suppressed by the environment
                                        itself. This flag allows force the reset.
        :return: the environment response as returned in get_last_env_response
        """
        [agent.reset_internal_state() for agent in self.agents.values()]
        self.reset_required = False
        if self.real_environment.current_episode_steps_counter == 0:
            self.last_env_response = self.real_environment.last_env_response
        return self.last_env_response

    @property
    def action_space(self) -> Dict[str, ActionSpace]:
        """
        Get the action space of each of the agents wrapped in this environment.
        :return: the action space
        """
        cagents_dict = self.agents
        cagents_names = cagents_dict.keys()

        return {name: cagents_dict[name].in_action_space for name in cagents_names}

    def get_random_action(self) -> Dict[str, ActionType]:
        """
        Get a random action from the environment action space
        :return: An action that follows the definition of the action space.
        """
        action_spaces = self.action_space  # The action spaces of the abstracted composite agents in this level
        return {name: action_space.sample() for name, action_space in action_spaces.items()}

    def get_random_action_with_info(self) -> Dict[str, ActionInfo]:
        """
        Get a random action from the environment action space and wrap it with additional info
        :return: An action that follows the definition of the action space with additional generated info.
        """
        return {k: ActionInfo(v) for k, v in self.get_random_action().items()}

    def build(self) -> None:
        """
        Build all the internal components of the level manager (composite agents and environment).
        :return: None
        """
        # TODO: move the spaces definition class to the environment?
        action_space = self.environment.action_space
        if isinstance(action_space, dict):  # TODO: shouldn't be a dict
            action_space = list(action_space.values())[0]
        spaces = SpacesDefinition(state=self.real_environment.state_space,
                                  goal=self.real_environment.goal_space,  # in HRL the agent might want to override this
                                  action=action_space,
                                  reward=self.real_environment.reward_space)
        [agent.set_environment_parameters(spaces) for agent in self.agents.values()]

    def setup_logger(self) -> None:
        """
        Setup the logger for all the agents in the level
        :return: None
        """
        [agent.setup_logger() for agent in self.agents.values()]

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the composite agents in the level manager
        :return: None
        """
        [agent.set_session(sess) for agent in self.agents.values()]

    def train(self) -> None:
        """
        Make a training step for all the composite agents in this level manager
        :return: the loss?
        """
        # both to screen and to csv
        [agent.train() for agent in self.agents.values()]

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the level manager
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the level manager and all the hierarchy levels below it
        :param val: the new phase
        :return: None
        """
        self._phase = val
        for agent in self.agents.values():
            agent.phase = val

    def step(self, action: Union[None, Dict[str, ActionType]]) -> EnvResponse:
        """
        Run a single step of following the behavioral scheme set for this environment.
        :param action: the action to apply to the agents held in this level, before beginning following
                       the scheme.
        :return: None
        """
        # set the incoming directive for the sub-agent (goal / skill selection / etc.)
        if action is not None:
            for agent_name, agent in self.agents.items():
                agent.set_incoming_directive(action)

        if self.reset_required:
            self.reset_internal_state()

        # get last response or initial response from the environment
        env_response = copy.copy(self.environment.last_env_response)

        # step for several time steps
        accumulated_reward = 0
        acting_agent = list(self.agents.values())[0]

        for i in range(self.steps_limit.num_steps):
            # let the agent observe the result and decide if it wants to terminate the episode
            done = acting_agent.observe(env_response)

            if done:
                break
            else:
                # get action
                action_info = acting_agent.act()

                # imitation agents will return no action since they don't play during training
                if action_info:
                    # step environment
                    env_response = self.environment.step(action_info.action)

                    # accumulate rewards such that the master policy will see the total reward during the step phase
                    accumulated_reward += env_response.reward

        # update the env response that will be exposed to the parent agent
        env_response_for_upper_level = copy.copy(env_response)
        env_response_for_upper_level.reward = accumulated_reward
        self.last_env_response = env_response_for_upper_level

        # if the environment terminated the episode -> let the agent observe the last response
        # in HRL,excluding top level one, we will always enter the below if clause
        # (because should_reset_agent_state_after_time_limit_passes is set to True)
        if env_response.game_over or self.should_reset_agent_state_after_time_limit_passes:
            # this is the agent's only opportunity to observe this transition - he will not get another one
            acting_agent.observe(env_response)  # TODO: acting agent? maybe all of the agents in the layer?
            self.handle_episode_ended()
            self.reset_required = True

        return env_response_for_upper_level

    def save_checkpoint(self, checkpoint_prefix: str) -> None:
        """
        Save checkpoints of the networks of all agents
        :param: checkpoint_prefix: The prefix of the checkpoint file to save
        :return: None
        """
        [agent.save_checkpoint(checkpoint_prefix) for agent in self.agents.values()]

    def restore_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Restores checkpoints of the networks of all agents
        :return: None
        """
        [agent.restore_checkpoint(checkpoint_dir) for agent in self.agents.values()]

    def sync(self) -> None:
        """
        Sync the networks of the agents with the global network parameters
        :return:
        """
        [agent.sync() for agent in self.agents.values()]

    def should_train(self) -> bool:
        return any([agent._should_train_helper() for agent in self.agents.values()])

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #               an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_step_on_trainer(self, transition: Transition) -> None:
        """
        This emulates a step using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Run a single step of following the behavioral scheme set for this environment.
        :param action: the action to apply to the agents held in this level, before beginning following
                       the scheme.
        :return: None
        """

        if self.reset_required:
            self.reset_internal_state()

        acting_agent = list(self.agents.values())[0]

        # for i in range(self.steps_limit.num_steps):
        # let the agent observe the result and decide if it wants to terminate the episode
        done = acting_agent.emulate_observe_on_trainer(transition)
        acting_agent.emulate_act_on_trainer(transition)

        if done:
            self.handle_episode_ended()
            self.reset_required = True

    def should_stop(self) -> bool:
        return all([agent.get_success_rate() >= self.environment.get_target_success_rate() for agent in self.agents.values()])

    def collect_savers(self) -> SaverCollection:
        """
        Calls collect_savers() on all agents and combines the results to a single collection
        :return: saver collection of all agent savers
        """
        savers = SaverCollection()
        for agent in self.agents.values():
            savers.update(agent.collect_savers(parent_path_suffix=self.name))
        return savers
