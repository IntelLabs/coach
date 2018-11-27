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
import os
import random
from collections import OrderedDict
from typing import Dict, List, Union, Tuple

import numpy as np
from pandas import read_pickle
from six.moves import range

from rl_coach.agents.agent_interface import AgentInterface
from rl_coach.architectures.network_wrapper import NetworkWrapper
from rl_coach.base_parameters import AgentParameters, Device, DeviceType, DistributedTaskParameters, Frameworks
from rl_coach.core_types import RunPhase, PredictionType, EnvironmentEpisodes, ActionType, Batch, Episode, StateType
from rl_coach.core_types import Transition, ActionInfo, TrainingSteps, EnvironmentSteps, EnvResponse
from rl_coach.logger import screen, Logger, EpisodeLogger
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplay
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition, VectorObservationSpace, GoalsSpace, AttentionActionSpace
from rl_coach.utils import Signal, force_list
from rl_coach.utils import dynamic_import_and_instantiate_module_from_params
from rl_coach.memories.backend.memory_impl import get_memory_backend


class Agent(AgentInterface):
    def __init__(self, agent_parameters: AgentParameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        """
        :param agent_parameters: A AgentParameters class instance with all the agent parameters
        """
        super().__init__()
        self.ap = agent_parameters
        self.task_id = self.ap.task_parameters.task_index
        self.is_chief = self.task_id == 0
        self.shared_memory = type(agent_parameters.task_parameters) == DistributedTaskParameters \
                             and self.ap.memory.shared_memory
        if self.shared_memory:
            self.shared_memory_scratchpad = self.ap.task_parameters.shared_memory_scratchpad
        self.name = agent_parameters.name
        self.parent = parent
        self.parent_level_manager = None
        self.full_name_id = agent_parameters.full_name_id = self.name

        if type(agent_parameters.task_parameters) == DistributedTaskParameters:
            screen.log_title("Creating agent - name: {} task id: {} (may take up to 30 seconds due to "
                             "tensorflow wake up time)".format(self.full_name_id, self.task_id))
        else:
            screen.log_title("Creating agent - name: {}".format(self.full_name_id))
        self.imitation = False
        self.agent_logger = Logger()
        self.agent_episode_logger = EpisodeLogger()

        # get the memory
        # - distributed training + shared memory:
        #   * is chief?  -> create the memory and add it to the scratchpad
        #   * not chief? -> wait for the chief to create the memory and then fetch it
        # - non distributed training / not shared memory:
        #   * create memory
        memory_name = self.ap.memory.path.split(':')[1]
        self.memory_lookup_name = self.full_name_id + '.' + memory_name
        if self.shared_memory and not self.is_chief:
            self.memory = self.shared_memory_scratchpad.get(self.memory_lookup_name)
        else:
            # modules
            self.memory = dynamic_import_and_instantiate_module_from_params(self.ap.memory)

            if hasattr(self.ap.memory, 'memory_backend_params'):
                self.memory_backend = get_memory_backend(self.ap.memory.memory_backend_params)

                if self.ap.memory.memory_backend_params.run_type != 'trainer':
                    self.memory.set_memory_backend(self.memory_backend)

            if agent_parameters.memory.load_memory_from_file_path:
                screen.log_title("Loading replay buffer from pickle. Pickle path: {}"
                                 .format(agent_parameters.memory.load_memory_from_file_path))
                self.memory.load(agent_parameters.memory.load_memory_from_file_path)

            if self.shared_memory and self.is_chief:
                self.shared_memory_scratchpad.add(self.memory_lookup_name, self.memory)

        # set devices
        if type(agent_parameters.task_parameters) == DistributedTaskParameters:
            self.has_global = True
            self.replicated_device = agent_parameters.task_parameters.device
            self.worker_device = "/job:worker/task:{}".format(self.task_id)
            if agent_parameters.task_parameters.use_cpu:
                self.worker_device += "/cpu:0"
            else:
                self.worker_device += "/device:GPU:0"
        else:
            self.has_global = False
            self.replicated_device = None
            if agent_parameters.task_parameters.use_cpu:
                self.worker_device = Device(DeviceType.CPU)
            else:
                self.worker_device = [Device(DeviceType.GPU, i)
                                      for i in range(agent_parameters.task_parameters.num_gpu)]

        # filters
        self.input_filter = self.ap.input_filter
        self.input_filter.set_name('input_filter')
        self.output_filter = self.ap.output_filter
        self.output_filter.set_name('output_filter')
        self.pre_network_filter = self.ap.pre_network_filter
        self.pre_network_filter.set_name('pre_network_filter')

        device = self.replicated_device if self.replicated_device else self.worker_device

        # TODO-REMOVE This is a temporary flow dividing to 3 modes. To be converged to a single flow once distributed tf
        #  is removed, and Redis is used for sharing data between local workers.
        # Filters MoW will be split between different configurations
        # 1. Distributed coach synchrnization type (=distributed across multiple nodes) - Redis based data sharing + numpy arithmetic backend
        # 2. Distributed TF (=distributed on a single node, using distributed TF) - TF for both data sharing and arithmetic backend
        # 3. Single worker (=both TF and Mxnet) - no data sharing needed + numpy arithmetic backend

        if hasattr(self.ap.memory, 'memory_backend_params') and self.ap.algorithm.distributed_coach_synchronization_type:
            self.input_filter.set_device(device, memory_backend_params=self.ap.memory.memory_backend_params, mode='numpy')
            self.output_filter.set_device(device, memory_backend_params=self.ap.memory.memory_backend_params, mode='numpy')
            self.pre_network_filter.set_device(device, memory_backend_params=self.ap.memory.memory_backend_params, mode='numpy')
        elif (type(agent_parameters.task_parameters) == DistributedTaskParameters and
              agent_parameters.task_parameters.framework_type == Frameworks.tensorflow):
            self.input_filter.set_device(device, mode='tf')
            self.output_filter.set_device(device, mode='tf')
            self.pre_network_filter.set_device(device, mode='tf')
        else:
            self.input_filter.set_device(device, mode='numpy')
            self.output_filter.set_device(device, mode='numpy')
            self.pre_network_filter.set_device(device, mode='numpy')

        # initialize all internal variables
        self._phase = RunPhase.HEATUP
        self.total_shaped_reward_in_current_episode = 0
        self.total_reward_in_current_episode = 0
        self.total_steps_counter = 0
        self.running_reward = None
        self.training_iteration = 0
        self.last_target_network_update_step = 0
        self.last_training_phase_step = 0
        self.current_episode = self.ap.current_episode = 0
        self.curr_state = {}
        self.current_hrl_goal = None
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        self.last_episode_evaluation_ran = 0
        self.running_observations = []
        self.agent_logger.set_current_time(self.current_episode)
        self.exploration_policy = None
        self.networks = {}
        self.last_action_info = None
        self.running_observation_stats = None
        self.running_reward_stats = None
        self.accumulated_rewards_across_evaluation_episodes = 0
        self.accumulated_shaped_rewards_across_evaluation_episodes = 0
        self.num_successes_across_evaluation_episodes = 0
        self.num_evaluation_episodes_completed = 0
        self.current_episode_buffer = Episode(discount=self.ap.algorithm.discount, n_step=self.ap.algorithm.n_step)
        # TODO: add agents observation rendering for debugging purposes (not the same as the environment rendering)

        # environment parameters
        self.spaces = None
        self.in_action_space = self.ap.algorithm.in_action_space

        # signals
        self.episode_signals = []
        self.step_signals = []
        self.loss = self.register_signal('Loss')
        self.curr_learning_rate = self.register_signal('Learning Rate')
        self.unclipped_grads = self.register_signal('Grads (unclipped)')
        self.reward = self.register_signal('Reward', dump_one_value_per_episode=False, dump_one_value_per_step=True)
        self.shaped_reward = self.register_signal('Shaped Reward', dump_one_value_per_episode=False, dump_one_value_per_step=True)
        self.discounted_return = self.register_signal('Discounted Return')
        if isinstance(self.in_action_space, GoalsSpace):
            self.distance_from_goal = self.register_signal('Distance From Goal', dump_one_value_per_step=True)
        # use seed
        if self.ap.task_parameters.seed is not None:
            random.seed(self.ap.task_parameters.seed)
            np.random.seed(self.ap.task_parameters.seed)
        else:
            # we need to seed the RNG since the different processes are initialized with the same parent seed
            random.seed()
            np.random.seed()

    @property
    def parent(self) -> 'LevelManager':
        """
        Get the parent class of the agent

        :return: the current phase
        """
        return self._parent

    @parent.setter
    def parent(self, val) -> None:
        """
        Change the parent class of the agent.
        Additionally, updates the full name of the agent

        :param val: the new parent
        :return: None
        """
        self._parent = val
        if self._parent is not None:
            if not hasattr(self._parent, 'name'):
                raise ValueError("The parent of an agent must have a name")
            self.full_name_id = self.ap.full_name_id = "{}/{}".format(self._parent.name, self.name)

    def setup_logger(self) -> None:
        """
        Setup the logger for the agent

        :return: None
        """
        # dump documentation
        logger_prefix = "{graph_name}.{level_name}.{agent_full_id}".\
            format(graph_name=self.parent_level_manager.parent_graph_manager.name,
                   level_name=self.parent_level_manager.name,
                   agent_full_id='.'.join(self.full_name_id.split('/')))
        self.agent_logger.set_logger_filenames(self.ap.task_parameters.experiment_path, logger_prefix=logger_prefix,
                                               add_timestamp=True, task_id=self.task_id)
        if self.ap.visualization.dump_in_episode_signals:
            self.agent_episode_logger.set_logger_filenames(self.ap.task_parameters.experiment_path,
                                                           logger_prefix=logger_prefix,
                                                           add_timestamp=True, task_id=self.task_id)

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the agents in the composite agent

        :return: None
        """
        self.input_filter.set_session(sess)
        self.output_filter.set_session(sess)
        self.pre_network_filter.set_session(sess)
        [network.set_session(sess) for network in self.networks.values()]

    def register_signal(self, signal_name: str, dump_one_value_per_episode: bool=True,
                        dump_one_value_per_step: bool=False) -> Signal:
        """
        Register a signal such that its statistics will be dumped and be viewable through dashboard

        :param signal_name: the name of the signal as it will appear in dashboard
        :param dump_one_value_per_episode: should the signal value be written for each episode?
        :param dump_one_value_per_step: should the signal value be written for each step?
        :return: the created signal
        """
        signal = Signal(signal_name)
        if dump_one_value_per_episode:
            self.episode_signals.append(signal)
        if dump_one_value_per_step:
            self.step_signals.append(signal)
        return signal

    def set_environment_parameters(self, spaces: SpacesDefinition):
        """
        Sets the parameters that are environment dependent. As a side effect, initializes all the components that are
        dependent on those values, by calling init_environment_dependent_modules

        :param spaces: the environment spaces definition
        :return: None
        """
        self.spaces = copy.deepcopy(spaces)

        if self.ap.algorithm.use_accumulated_reward_as_measurement:
            if 'measurements' in self.spaces.state.sub_spaces:
                self.spaces.state['measurements'].shape += 1
                self.spaces.state['measurements'].measurements_names += ['accumulated_reward']
            else:
                self.spaces.state['measurements'] = VectorObservationSpace(1, measurements_names=['accumulated_reward'])

        for observation_name in self.spaces.state.sub_spaces.keys():
            self.spaces.state[observation_name] = \
                self.pre_network_filter.get_filtered_observation_space(observation_name,
                    self.input_filter.get_filtered_observation_space(observation_name,
                                                                     self.spaces.state[observation_name]))

        self.spaces.reward = self.pre_network_filter.get_filtered_reward_space(
            self.input_filter.get_filtered_reward_space(self.spaces.reward))

        self.spaces.action = self.output_filter.get_unfiltered_action_space(self.spaces.action)

        if isinstance(self.in_action_space, GoalsSpace):
            # TODO: what if the goal type is an embedding / embedding change?
            self.spaces.goal = self.in_action_space
            self.spaces.goal.set_target_space(self.spaces.state[self.spaces.goal.goal_name])

        self.init_environment_dependent_modules()

    def create_networks(self) -> Dict[str, NetworkWrapper]:
        """
        Create all the networks of the agent.
        The network creation will be done after setting the environment parameters for the agent, since they are needed
        for creating the network.

        :return: A list containing all the networks
        """
        networks = {}
        for network_name in sorted(self.ap.network_wrappers.keys()):
            networks[network_name] = NetworkWrapper(name=network_name,
                                                    agent_parameters=self.ap,
                                                    has_target=self.ap.network_wrappers[network_name].create_target_network,
                                                    has_global=self.has_global,
                                                    spaces=self.spaces,
                                                    replicated_device=self.replicated_device,
                                                    worker_device=self.worker_device)

            if self.ap.visualization.print_networks_summary:
                print(networks[network_name])

        return networks

    def init_environment_dependent_modules(self) -> None:
        """
        Initialize any modules that depend on knowing information about the environment such as the action space or
        the observation space

        :return: None
        """
        # initialize exploration policy
        if isinstance(self.ap.exploration, dict):
            if self.spaces.action.__class__ in self.ap.exploration.keys():
                self.ap.exploration = self.ap.exploration[self.spaces.action.__class__]
            else:
                raise ValueError("The exploration parameters were defined as a mapping between action space types and "
                                 "exploration types, but the action space used by the environment ({}) was not part of "
                                 "the exploration parameters dictionary keys ({})"
                                 .format(self.spaces.action.__class__, list(self.ap.exploration.keys())))
        self.ap.exploration.action_space = self.spaces.action
        self.exploration_policy = dynamic_import_and_instantiate_module_from_params(self.ap.exploration)

        # create all the networks of the agent
        self.networks = self.create_networks()

    @property
    def phase(self) -> RunPhase:
        """
        The current running phase of the agent

        :return: RunPhase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase) -> None:
        """
        Change the phase of the run for the agent and all the sub components

        :param val: the new run phase (TRAIN, TEST, etc.)
        :return: None
        """
        self.reset_evaluation_state(val)
        self._phase = val
        self.exploration_policy.change_phase(val)

    def reset_evaluation_state(self, val: RunPhase) -> None:
        """
        Perform accumulators initialization when entering an evaluation phase, and signal dumping when exiting an
        evaluation phase. Entering or exiting the evaluation phase is determined according to the new phase given
        by val, and by the current phase set in self.phase.

        :param val: The new phase to change to
        :return: None
        """
        starting_evaluation = (val == RunPhase.TEST)
        ending_evaluation = (self.phase == RunPhase.TEST)

        if starting_evaluation:
            self.accumulated_rewards_across_evaluation_episodes = 0
            self.accumulated_shaped_rewards_across_evaluation_episodes = 0
            self.num_successes_across_evaluation_episodes = 0
            self.num_evaluation_episodes_completed = 0
            if self.ap.is_a_highest_level_agent or self.ap.task_parameters.verbosity == "high":
                screen.log_title("{}: Starting evaluation phase".format(self.name))

        elif ending_evaluation:
            # we write to the next episode, because it could be that the current episode was already written
            # to disk and then we won't write it again
            self.agent_logger.set_current_time(self.current_episode + 1)
            evaluation_reward = self.accumulated_rewards_across_evaluation_episodes / self.num_evaluation_episodes_completed
            self.agent_logger.create_signal_value(
                'Evaluation Reward', evaluation_reward)
            self.agent_logger.create_signal_value(
                'Shaped Evaluation Reward',
                self.accumulated_shaped_rewards_across_evaluation_episodes / self.num_evaluation_episodes_completed)
            success_rate = self.num_successes_across_evaluation_episodes / self.num_evaluation_episodes_completed
            self.agent_logger.create_signal_value(
                "Success Rate",
                success_rate
            )
            if self.ap.is_a_highest_level_agent or self.ap.task_parameters.verbosity == "high":
                screen.log_title("{}: Finished evaluation phase. Success rate = {}, Avg Total Reward = {}"
                                 .format(self.name, np.round(success_rate, 2), np.round(evaluation_reward, 2)))

    def call_memory(self, func, args=()):
        """
        This function is a wrapper to allow having the same calls for shared or unshared memories.
        It should be used instead of calling the memory directly in order to allow different algorithms to work
        both with a shared and a local memory.

        :param func: the name of the memory function to call
        :param args: the arguments to supply to the function
        :return: the return value of the function
        """
        if self.shared_memory:
            result = self.shared_memory_scratchpad.internal_call(self.memory_lookup_name, func, args)
        else:
            if type(args) != tuple:
                args = (args,)
            result = getattr(self.memory, func)(*args)
        return result

    def log_to_screen(self) -> None:
        """
        Write an episode summary line to the terminal

        :return: None
        """
        # log to screen
        log = OrderedDict()
        log["Name"] = self.full_name_id
        if self.task_id is not None:
            log["Worker"] = self.task_id
        log["Episode"] = self.current_episode
        log["Total reward"] = np.round(self.total_reward_in_current_episode, 2)
        log["Exploration"] = np.round(self.exploration_policy.get_control_param(), 2)
        log["Steps"] = self.total_steps_counter
        log["Training iteration"] = self.training_iteration
        screen.log_dict(log, prefix=self.phase.value)

    def update_step_in_episode_log(self) -> None:
        """
        Updates the in-episode log file with all the signal values from the most recent step.

        :return: None
        """
        # log all the signals to file
        self.agent_episode_logger.set_current_time(self.current_episode_steps_counter)
        self.agent_episode_logger.create_signal_value('Training Iter', self.training_iteration)
        self.agent_episode_logger.create_signal_value('In Heatup', int(self._phase == RunPhase.HEATUP))
        self.agent_episode_logger.create_signal_value('ER #Transitions', self.call_memory('num_transitions'))
        self.agent_episode_logger.create_signal_value('ER #Episodes', self.call_memory('length'))
        self.agent_episode_logger.create_signal_value('Total steps', self.total_steps_counter)
        self.agent_episode_logger.create_signal_value("Epsilon", self.exploration_policy.get_control_param())
        self.agent_episode_logger.create_signal_value("Shaped Accumulated Reward", self.total_shaped_reward_in_current_episode)
        self.agent_episode_logger.create_signal_value('Update Target Network', 0, overwrite=False)
        self.agent_episode_logger.update_wall_clock_time(self.current_episode_steps_counter)

        for signal in self.step_signals:
            self.agent_episode_logger.create_signal_value(signal.name, signal.get_last_value())

        # dump
        self.agent_episode_logger.dump_output_csv()

    def update_log(self) -> None:
        """
        Updates the episodic log file with all the signal values from the most recent episode.
        Additional signals for logging can be set by the creating a new signal using self.register_signal,
        and then updating it with some internal agent values.

        :return: None
        """
        # log all the signals to file
        self.agent_logger.set_current_time(self.current_episode)
        self.agent_logger.create_signal_value('Training Iter', self.training_iteration)
        self.agent_logger.create_signal_value('In Heatup', int(self._phase == RunPhase.HEATUP))
        self.agent_logger.create_signal_value('ER #Transitions', self.call_memory('num_transitions'))
        self.agent_logger.create_signal_value('ER #Episodes', self.call_memory('length'))
        self.agent_logger.create_signal_value('Episode Length', self.current_episode_steps_counter)
        self.agent_logger.create_signal_value('Total steps', self.total_steps_counter)
        self.agent_logger.create_signal_value("Epsilon", np.mean(self.exploration_policy.get_control_param()))
        self.agent_logger.create_signal_value("Shaped Training Reward", self.total_shaped_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)
        self.agent_logger.create_signal_value("Training Reward", self.total_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)

        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)
        self.agent_logger.update_wall_clock_time(self.current_episode)

        if self._phase != RunPhase.TEST:
            self.agent_logger.create_signal_value('Evaluation Reward', np.nan, overwrite=False)
            self.agent_logger.create_signal_value('Shaped Evaluation Reward', np.nan, overwrite=False)
            self.agent_logger.create_signal_value('Success Rate', np.nan, overwrite=False)

        for signal in self.episode_signals:
            self.agent_logger.create_signal_value("{}/Mean".format(signal.name), signal.get_mean())
            self.agent_logger.create_signal_value("{}/Stdev".format(signal.name), signal.get_stdev())
            self.agent_logger.create_signal_value("{}/Max".format(signal.name), signal.get_max())
            self.agent_logger.create_signal_value("{}/Min".format(signal.name), signal.get_min())

        # dump
        if self.current_episode % self.ap.visualization.dump_signals_to_csv_every_x_episodes == 0 \
                and self.current_episode > 0:
            self.agent_logger.dump_output_csv()

    def handle_episode_ended(self) -> None:
        """
        Make any changes needed when each episode is ended.
        This includes incrementing counters, updating full episode dependent values, updating logs, etc.
        This function is called right after each episode is ended.

        :return: None
        """
        self.current_episode_buffer.is_complete = True
        self.current_episode_buffer.update_transitions_rewards_and_bootstrap_data()

        for transition in self.current_episode_buffer.transitions:
            self.discounted_return.add_sample(transition.n_step_discounted_rewards)

        if self.phase != RunPhase.TEST or self.ap.task_parameters.evaluate_only:
            self.current_episode += 1

        if self.phase != RunPhase.TEST:
            if isinstance(self.memory, EpisodicExperienceReplay):
                self.call_memory('store_episode', self.current_episode_buffer)
            elif self.ap.algorithm.store_transitions_only_when_episodes_are_terminated:
                for transition in self.current_episode_buffer.transitions:
                    self.call_memory('store', transition)

        if self.phase == RunPhase.TEST:
            self.accumulated_rewards_across_evaluation_episodes += self.total_reward_in_current_episode
            self.accumulated_shaped_rewards_across_evaluation_episodes += self.total_shaped_reward_in_current_episode
            self.num_evaluation_episodes_completed += 1

            if self.spaces.reward.reward_success_threshold and \
                    self.total_reward_in_current_episode >= self.spaces.reward.reward_success_threshold:
                self.num_successes_across_evaluation_episodes += 1

        if self.ap.visualization.dump_csv:
            self.update_log()

        if self.ap.is_a_highest_level_agent or self.ap.task_parameters.verbosity == "high":
            self.log_to_screen()

    def reset_internal_state(self) -> None:
        """
        Reset all the episodic parameters. This function is called right before each episode starts.

        :return: None
        """
        for signal in self.episode_signals:
            signal.reset()
        for signal in self.step_signals:
            signal.reset()
        self.agent_episode_logger.set_episode_idx(self.current_episode)
        self.total_shaped_reward_in_current_episode = 0
        self.total_reward_in_current_episode = 0
        self.curr_state = {}
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        self.current_episode_buffer = Episode(discount=self.ap.algorithm.discount, n_step=self.ap.algorithm.n_step)
        if self.exploration_policy:
            self.exploration_policy.reset()
        self.input_filter.reset()
        self.output_filter.reset()
        self.pre_network_filter.reset()
        if isinstance(self.memory, EpisodicExperienceReplay):
            self.call_memory('verify_last_episode_is_closed')

        for network in self.networks.values():
            network.online_network.reset_internal_memory()

    def learn_from_batch(self, batch) -> Tuple[float, List, List]:
        """
        Given a batch of transitions, calculates their target values and updates the network.

        :param batch: A list of transitions
        :return: The total loss of the training, the loss per head and the unclipped gradients
        """
        return 0, [], []

    def _should_update_online_weights_to_target(self):
        """
        Determine if online weights should be copied to the target.

        :return: boolean: True if the online weights should be copied to the target.
        """

        # update the target network of every network that has a target network
        step_method = self.ap.algorithm.num_steps_between_copying_online_weights_to_target
        if step_method.__class__ == TrainingSteps:
            should_update = (self.training_iteration - self.last_target_network_update_step) >= step_method.num_steps
            if should_update:
                self.last_target_network_update_step = self.training_iteration
        elif step_method.__class__ == EnvironmentSteps:
            should_update = (self.total_steps_counter - self.last_target_network_update_step) >= step_method.num_steps
            if should_update:
                self.last_target_network_update_step = self.total_steps_counter
        else:
            raise ValueError("The num_steps_between_copying_online_weights_to_target parameter should be either "
                             "EnvironmentSteps or TrainingSteps. Instead it is {}".format(step_method.__class__))
        return should_update

    def _should_train(self):
        """
        Determine if we should start a training phase according to the number of steps passed since the last training

        :return:  boolean: True if we should start a training phase
        """

        should_update = self._should_train_helper()

        step_method = self.ap.algorithm.num_consecutive_playing_steps

        if should_update:
            if step_method.__class__ == EnvironmentEpisodes:
                self.last_training_phase_step = self.current_episode
            if step_method.__class__ == EnvironmentSteps:
                self.last_training_phase_step = self.total_steps_counter

        return should_update

    def _should_train_helper(self):
        wait_for_full_episode = self.ap.algorithm.act_for_full_episodes
        step_method = self.ap.algorithm.num_consecutive_playing_steps

        if step_method.__class__ == EnvironmentEpisodes:
            should_update = (self.current_episode - self.last_training_phase_step) >= step_method.num_steps
            should_update = should_update and self.call_memory('length') > 0

        elif step_method.__class__ == EnvironmentSteps:
            should_update = (self.total_steps_counter - self.last_training_phase_step) >= step_method.num_steps
            should_update = should_update and self.call_memory('num_transitions') > 0

            if wait_for_full_episode:
                should_update = should_update and self.current_episode_buffer.is_complete
        else:
            raise ValueError("The num_consecutive_playing_steps parameter should be either "
                             "EnvironmentSteps or Episodes. Instead it is {}".format(step_method.__class__))

        return should_update

    def train(self) -> float:
        """
        Check if a training phase should be done as configured by num_consecutive_playing_steps.
        If it should, then do several training steps as configured by num_consecutive_training_steps.
        A single training iteration: Sample a batch, train on it and update target networks.

        :return: The total training loss during the training iterations.
        """
        loss = 0
        if self._should_train():
            for network in self.networks.values():
                network.set_is_training(True)

            for training_step in range(self.ap.algorithm.num_consecutive_training_steps):
                # TODO: this should be network dependent
                network_parameters = list(self.ap.network_wrappers.values())[0]

                # update counters
                self.training_iteration += 1

                # sample a batch and train on it
                batch = self.call_memory('sample', network_parameters.batch_size)
                if self.pre_network_filter is not None:
                    batch = self.pre_network_filter.filter(batch, update_internal_state=False, deep_copy=False)

                # if the batch returned empty then there are not enough samples in the replay buffer -> skip
                # training step
                if len(batch) > 0:
                    # train
                    batch = Batch(batch)
                    total_loss, losses, unclipped_grads = self.learn_from_batch(batch)
                    loss += total_loss
                    self.unclipped_grads.add_sample(unclipped_grads)

                    # TODO: the learning rate decay should be done through the network instead of here
                    # decay learning rate
                    if network_parameters.learning_rate_decay_rate != 0:
                        self.curr_learning_rate.add_sample(self.networks['main'].sess.run(
                            self.networks['main'].online_network.current_learning_rate))
                    else:
                        self.curr_learning_rate.add_sample(network_parameters.learning_rate)

                    if any([network.has_target for network in self.networks.values()]) \
                            and self._should_update_online_weights_to_target():
                        for network in self.networks.values():
                            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

                        self.agent_logger.create_signal_value('Update Target Network', 1)
                    else:
                        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

                    self.loss.add_sample(loss)

                    if self.imitation:
                        self.log_to_screen()

            for network in self.networks.values():
                network.set_is_training(False)

            # run additional commands after the training is done
            self.post_training_commands()

        return loss

    def choose_action(self, curr_state):
        """
        choose an action to act with in the current episode being played. Different behavior might be exhibited when
        training or testing.

        :param curr_state: the current state to act upon.
        :return: chosen action, some action value describing the action (q-value, probability, etc)
        """
        pass

    def prepare_batch_for_inference(self, states: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]],
                                    network_name: str) -> Dict[str, np.array]:
        """
        Convert curr_state into input tensors tensorflow is expecting. i.e. if we have several inputs states, stack all
        observations together, measurements together, etc.

        :param states: A list of environment states, where each one is a dict mapping from an observation name to its
                       corresponding observation
        :param network_name: The agent network name to prepare the batch for. this is needed in order to extract only
                             the observation relevant for the network from the states.
        :return: A dictionary containing a list of values from all the given states for each of the observations
        """
        # convert to batch so we can run it through the network
        states = force_list(states)
        batches_dict = {}
        for key in self.ap.network_wrappers[network_name].input_embedders_parameters.keys():
            # there are cases (e.g. ddpg) where the state does not contain all the information needed for running
            # through the network and this has to be added externally (e.g. ddpg where the action needs to be given in
            # addition to the current_state, so that all the inputs of the network will be filled)
            if key in states[0].keys():
                batches_dict[key] = np.array([np.array(state[key]) for state in states])

        return batches_dict

    def act(self) -> ActionInfo:
        """
        Given the agents current knowledge, decide on the next action to apply to the environment

        :return: An ActionInfo object, which contains the action and any additional info from the action decision process
        """
        if self.phase == RunPhase.TRAIN and self.ap.algorithm.num_consecutive_playing_steps.num_steps == 0:
            # This agent never plays  while training (e.g. behavioral cloning)
            return None

        # count steps (only when training or if we are in the evaluation worker)
        if self.phase != RunPhase.TEST or self.ap.task_parameters.evaluate_only:
            self.total_steps_counter += 1
        self.current_episode_steps_counter += 1

        # decide on the action
        if self.phase == RunPhase.HEATUP and not self.ap.algorithm.heatup_using_network_decisions:
            # random action
            self.last_action_info = self.spaces.action.sample_with_info()
        else:
            # informed action
            if self.pre_network_filter is not None:
                # before choosing an action, first use the pre_network_filter to filter out the current state
                curr_state = self.run_pre_network_filter_for_inference(self.curr_state)

            else:
                curr_state = self.curr_state
            self.last_action_info = self.choose_action(curr_state)

        filtered_action_info = self.output_filter.filter(self.last_action_info)

        return filtered_action_info

    def run_pre_network_filter_for_inference(self, state: StateType) -> StateType:
        """
        Run filters which where defined for being applied right before using the state for inference.

        :param state: The state to run the filters on
        :return: The filtered state
        """
        dummy_env_response = EnvResponse(next_state=state, reward=0, game_over=False)
        return self.pre_network_filter.filter(dummy_env_response)[0].next_state

    def get_state_embedding(self, state: dict) -> np.ndarray:
        """
        Given a state, get the corresponding state embedding  from the main network

        :param state: a state dict
        :return: a numpy embedding vector
        """
        # TODO: this won't work anymore
        # TODO: instead of the state embedding (which contains the goal) we should use the observation embedding
        embedding = self.networks['main'].online_network.predict(
            self.prepare_batch_for_inference(state, "main"),
            outputs=self.networks['main'].online_network.state_embedding)
        return embedding

    def update_transition_before_adding_to_replay_buffer(self, transition: Transition) -> Transition:
        """
        Allows agents to update the transition just before adding it to the replay buffer.
        Can be useful for agents that want to tweak the reward, termination signal, etc.

        :param transition: the transition to update
        :return: the updated transition
        """
        return transition

    def observe(self, env_response: EnvResponse) -> bool:
        """
        Given a response from the environment, distill the observation from it and store it for later use.
        The response should be a dictionary containing the performed action, the new observation and measurements,
        the reward, a game over flag and any additional information necessary.

        :param env_response: result of call from environment.step(action)
        :return: a boolean value which determines if the agent has decided to terminate the episode after seeing the
                 given observation
        """

        # filter the env_response
        filtered_env_response = self.input_filter.filter(env_response)[0]

        # inject agent collected statistics, if required
        if self.ap.algorithm.use_accumulated_reward_as_measurement:
            if 'measurements' in filtered_env_response.next_state:
                filtered_env_response.next_state['measurements'] = np.append(filtered_env_response.next_state['measurements'],
                                                                             self.total_shaped_reward_in_current_episode)
            else:
                filtered_env_response.next_state['measurements'] = np.array([self.total_shaped_reward_in_current_episode])

        # if we are in the first step in the episode, then we don't have a a next state and a reward and thus no
        # transition yet, and therefore we don't need to store anything in the memory.
        # also we did not reach the goal yet.
        if self.current_episode_steps_counter == 0:
            # initialize the current state
            self.curr_state = filtered_env_response.next_state
            return env_response.game_over
        else:
            transition = Transition(state=copy.copy(self.curr_state), action=self.last_action_info.action,
                                    reward=filtered_env_response.reward, next_state=filtered_env_response.next_state,
                                    game_over=filtered_env_response.game_over, info=filtered_env_response.info)

            # now that we have formed a basic transition - the next state progresses to be the current state
            self.curr_state = filtered_env_response.next_state

            # make agent specific changes to the transition if needed
            transition = self.update_transition_before_adding_to_replay_buffer(transition)

            # merge the intrinsic reward in
            if self.ap.algorithm.scale_external_reward_by_intrinsic_reward_value:
                transition.reward = transition.reward * (1 + self.last_action_info.action_intrinsic_reward)
            else:
                transition.reward = transition.reward + self.last_action_info.action_intrinsic_reward

            # sum up the total shaped reward
            self.total_shaped_reward_in_current_episode += transition.reward
            self.total_reward_in_current_episode += env_response.reward
            self.shaped_reward.add_sample(transition.reward)
            self.reward.add_sample(env_response.reward)

            # add action info to transition
            if type(self.parent).__name__ == 'CompositeAgent':
                transition.add_info(self.parent.last_action_info.__dict__)
            else:
                transition.add_info(self.last_action_info.__dict__)

            # create and store the transition
            if self.phase in [RunPhase.TRAIN, RunPhase.HEATUP]:
                # for episodic memories we keep the transitions in a local buffer until the episode is ended.
                # for regular memories we insert the transitions directly to the memory
                self.current_episode_buffer.insert(transition)
                if not isinstance(self.memory, EpisodicExperienceReplay) \
                        and not self.ap.algorithm.store_transitions_only_when_episodes_are_terminated:
                    self.call_memory('store', transition)

            if self.ap.visualization.dump_in_episode_signals:
                self.update_step_in_episode_log()

            return transition.game_over

    def post_training_commands(self) -> None:
        """
        A function which allows adding any functionality that is required to run right after the training phase ends.

        :return: None
        """
        pass

    def get_predictions(self, states: List[Dict[str, np.ndarray]], prediction_type: PredictionType):
        """
        Get a prediction from the agent with regard to the requested prediction_type.
        If the agent cannot predict this type of prediction_type, or if there is more than possible way to do so,
        raise a ValueException.

        :param states: The states to get a prediction for
        :param prediction_type: The type of prediction to get for the states. For example, the state-value prediction.
        :return: the predicted values
        """

        predictions = self.networks['main'].online_network.predict_with_prediction_type(
            # states=self.dict_state_to_batches_dict(states, 'main'), prediction_type=prediction_type)
            states=states, prediction_type=prediction_type)

        if len(predictions.keys()) != 1:
            raise ValueError("The network has more than one component {} matching the requested prediction_type {}. ".
                             format(list(predictions.keys()), prediction_type))
        return list(predictions.values())[0]

    def set_incoming_directive(self, action: ActionType) -> None:
        """
        Allows setting a directive for the agent to follow. This is useful in hierarchy structures, where the agent
        has another master agent that is controlling it. In such cases, the master agent can define the goals for the
        slave agent, define it's observation, possible actions, etc. The directive type is defined by the agent
        in-action-space.

        :param action: The action that should be set as the directive
        :return:
        """
        if isinstance(self.in_action_space, GoalsSpace):
            self.current_hrl_goal = action
        elif isinstance(self.in_action_space, AttentionActionSpace):
            self.input_filter.observation_filters['attention'].crop_low = action[0]
            self.input_filter.observation_filters['attention'].crop_high = action[1]
            self.output_filter.action_filters['masking'].set_masking(action[0], action[1])

    def save_checkpoint(self, checkpoint_prefix: str) -> None:
        """
        Allows agents to store additional information when saving checkpoints.

        :param checkpoint_prefix: The prefix of the checkpoint file to save
        :return: None
        """
        checkpoint_dir = self.ap.task_parameters.checkpoint_save_dir

        checkpoint_prefix = '.'.join([checkpoint_prefix] + self.full_name_id.split('/'))  # adds both level name and agent name

        self.input_filter.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)
        self.output_filter.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)
        self.pre_network_filter.save_state_to_checkpoint(checkpoint_dir, checkpoint_prefix)

    def restore_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Allows agents to store additional information when saving checkpoints.

        :param checkpoint_dir: The checkpoint dir to restore from
        :return: None
        """
        checkpoint_prefix = '.'.join(self.full_name_id.split('/'))  # adds both level name and agent name
        self.input_filter.restore_state_from_checkpoint(checkpoint_dir, checkpoint_prefix)
        self.pre_network_filter.restore_state_from_checkpoint(checkpoint_dir, checkpoint_prefix)

        # no output filters currently have an internal state to restore
        # self.output_filter.restore_state_from_checkpoint(checkpoint_dir)

    def sync(self) -> None:
        """
        Sync the global network parameters to local networks

        :return: None
        """
        for network in self.networks.values():
            network.sync()

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #               an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_observe_on_trainer(self, transition: Transition) -> bool:
        """
        This emulates the observe using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Given a response from the environment, distill the observation from it and store it for later use.
        The response should be a dictionary containing the performed action, the new observation and measurements,
        the reward, a game over flag and any additional information necessary.
        :return:
        """

        # if we are in the first step in the episode, then we don't have a a next state and a reward and thus no
        # transition yet, and therefore we don't need to store anything in the memory.
        # also we did not reach the goal yet.
        if self.current_episode_steps_counter == 0:
            # initialize the current state
            return transition.game_over
        else:
            # sum up the total shaped reward
            self.total_shaped_reward_in_current_episode += transition.reward
            self.total_reward_in_current_episode += transition.reward
            self.shaped_reward.add_sample(transition.reward)
            self.reward.add_sample(transition.reward)

            # create and store the transition
            if self.phase in [RunPhase.TRAIN, RunPhase.HEATUP]:
                # for episodic memories we keep the transitions in a local buffer until the episode is ended.
                # for regular memories we insert the transitions directly to the memory
                self.current_episode_buffer.insert(transition)
                if not isinstance(self.memory, EpisodicExperienceReplay) \
                        and not self.ap.algorithm.store_transitions_only_when_episodes_are_terminated:
                    self.call_memory('store', transition)

            if self.ap.visualization.dump_in_episode_signals:
                self.update_step_in_episode_log()

            return transition.game_over

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #         an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_act_on_trainer(self, transition: Transition) -> ActionInfo:
        """
        This emulates the act using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Given the agents current knowledge, decide on the next action to apply to the environment
        :return: an action and a dictionary containing any additional info from the action decision process
        """
        if self.phase == RunPhase.TRAIN and self.ap.algorithm.num_consecutive_playing_steps.num_steps == 0:
            # This agent never plays  while training (e.g. behavioral cloning)
            return None

        # count steps (only when training or if we are in the evaluation worker)
        if self.phase != RunPhase.TEST or self.ap.task_parameters.evaluate_only:
            self.total_steps_counter += 1
        self.current_episode_steps_counter += 1

        self.last_action_info = transition.action

        return self.last_action_info

    def get_success_rate(self) -> float:
        return self.num_successes_across_evaluation_episodes / self.num_evaluation_episodes_completed

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collect all of agent's network savers
        :param parent_path_suffix: path suffix of the parent of the agent
            (could be name of level manager or composite agent)
        :return: collection of all agent savers
        """
        parent_path_suffix = "{}.{}".format(parent_path_suffix, self.name)
        savers = SaverCollection()
        for network in self.networks.values():
            savers.update(network.collect_savers(parent_path_suffix))
        return savers
