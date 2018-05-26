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

import scipy.ndimage
try:
    import matplotlib.pyplot as plt
except:
    from logger import failed_imports
    failed_imports.append("matplotlib")

import copy
from renderer import Renderer
from configurations import Preset
from collections import deque
from utils import LazyStack
from collections import OrderedDict
from utils import RunPhase, Signal, is_empty, RunningStat
from architectures import *
from exploration_policies import *
from memories import *
from memories.memory import *
from logger import logger, screen
import random
import time
import os
import itertools
from architectures.tensorflow_components.shared_variables import SharedRunningStats
from six.moves import range


class Agent(object):
    def __init__(self, env, tuning_parameters, replicated_device=None, task_id=0):
        """
        :param env: An environment instance
        :type env: EnvironmentWrapper
        :param tuning_parameters: A Preset class instance with all the running paramaters
        :type tuning_parameters: Preset
        :param replicated_device: A tensorflow device for distributed training (optional)
        :type replicated_device: instancemethod
        :param thread_id: The current thread id
        :param thread_id: int
        """

        screen.log_title("Creating agent {}".format(task_id))
        self.task_id = task_id
        self.sess = tuning_parameters.sess
        self.env = tuning_parameters.env_instance = env
        self.imitation = False

        # i/o dimensions
        if not tuning_parameters.env.desired_observation_width or not tuning_parameters.env.desired_observation_height:
            tuning_parameters.env.desired_observation_width = self.env.width
            tuning_parameters.env.desired_observation_height = self.env.height
        self.action_space_size = tuning_parameters.env.action_space_size = self.env.action_space_size
        self.measurements_size = tuning_parameters.env.measurements_size = self.env.measurements_size
        if tuning_parameters.agent.use_accumulated_reward_as_measurement:
            self.measurements_size = tuning_parameters.env.measurements_size = (self.measurements_size[0] + 1,)

        # modules
        if tuning_parameters.agent.load_memory_from_file_path:
            screen.log_title("Loading replay buffer from pickle. Pickle path: {}"
                             .format(tuning_parameters.agent.load_memory_from_file_path))
            self.memory = read_pickle(tuning_parameters.agent.load_memory_from_file_path)
        else:
            self.memory = eval(tuning_parameters.memory + '(tuning_parameters)')
        # self.architecture = eval(tuning_parameters.architecture)

        self.has_global = replicated_device is not None
        self.replicated_device = replicated_device
        self.worker_device = "/job:worker/task:{}/cpu:0".format(task_id) if replicated_device is not None else "/gpu:0"

        self.exploration_policy = eval(tuning_parameters.exploration.policy + '(tuning_parameters)')
        self.evaluation_exploration_policy = eval(tuning_parameters.exploration.evaluation_policy
                                                  + '(tuning_parameters)')
        self.evaluation_exploration_policy.change_phase(RunPhase.TEST)

        # initialize all internal variables
        self.tp = tuning_parameters
        self.in_heatup = False
        self.total_reward_in_current_episode = 0
        self.total_steps_counter = 0
        self.running_reward = None
        self.training_iteration = 0
        self.current_episode = self.tp.current_episode = 0
        self.curr_state = {}
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        self.last_episode_evaluation_ran = 0
        self.running_observations = []
        logger.set_current_time(self.current_episode)
        self.main_network = None
        self.networks = []
        self.last_episode_images = []
        self.renderer = Renderer()

        # signals
        self.signals = []
        self.loss = Signal('Loss')
        self.signals.append(self.loss)
        self.curr_learning_rate = Signal('Learning Rate')
        self.signals.append(self.curr_learning_rate)

        if self.tp.env.normalize_observation and not self.env.is_state_type_image:
            if not self.tp.distributed or not self.tp.agent.share_statistics_between_workers:
                self.running_observation_stats = RunningStat((self.tp.env.desired_observation_width,))
                self.running_reward_stats = RunningStat(())
                if self.tp.checkpoint_restore_dir:
                    checkpoint_path = os.path.join(self.tp.checkpoint_restore_dir, "running_stats.p")
                    self.running_observation_stats = read_pickle(checkpoint_path)
                else:
                    self.running_observation_stats = RunningStat((self.tp.env.desired_observation_width,))
                    self.running_reward_stats = RunningStat(())
            else:
                self.running_observation_stats = SharedRunningStats(self.tp, replicated_device,
                                                                    shape=(self.tp.env.desired_observation_width,),
                                                                    name='observation_stats')
                self.running_reward_stats = SharedRunningStats(self.tp, replicated_device,
                                                               shape=(),
                                                               name='reward_stats')

        # env is already reset at this point. Otherwise we're getting an error where you cannot
        # reset an env which is not done
        self.reset_game(do_not_reset_env=True)

        # use seed
        if self.tp.seed is not None:
            random.seed(self.tp.seed)
            np.random.seed(self.tp.seed)

    def log_to_screen(self, phase):
        # log to screen
        if self.current_episode >= 0:
            if phase == RunPhase.TRAIN:
                exploration = self.exploration_policy.get_control_param()
            else:
                exploration = self.evaluation_exploration_policy.get_control_param()

            screen.log_dict(
                OrderedDict([
                    ("Worker", self.task_id),
                    ("Episode", self.current_episode),
                    ("total reward", self.total_reward_in_current_episode),
                    ("exploration", exploration),
                    ("steps", self.total_steps_counter),
                    ("training iteration", self.training_iteration)
                ]),
                prefix=phase
            )

    def update_log(self, phase=RunPhase.TRAIN):
        """
        Writes logging messages to screen and updates the log file with all the signal values.
        :return: None
        """
        # log all the signals to file
        logger.set_current_time(self.current_episode)
        logger.create_signal_value('Training Iter', self.training_iteration)
        logger.create_signal_value('In Heatup', int(phase == RunPhase.HEATUP))
        logger.create_signal_value('ER #Transitions', self.memory.num_transitions())
        logger.create_signal_value('ER #Episodes', self.memory.length())
        logger.create_signal_value('Episode Length', self.current_episode_steps_counter)
        logger.create_signal_value('Total steps', self.total_steps_counter)
        logger.create_signal_value("Epsilon", self.exploration_policy.get_control_param())
        logger.create_signal_value("Training Reward", self.total_reward_in_current_episode
                                   if phase == RunPhase.TRAIN else np.nan)
        logger.create_signal_value('Evaluation Reward', self.total_reward_in_current_episode
                                   if phase == RunPhase.TEST else np.nan)
        logger.create_signal_value('Update Target Network', 0, overwrite=False)
        logger.update_wall_clock_time(self.current_episode)

        for signal in self.signals:
            logger.create_signal_value("{}/Mean".format(signal.name), signal.get_mean())
            logger.create_signal_value("{}/Stdev".format(signal.name), signal.get_stdev())
            logger.create_signal_value("{}/Max".format(signal.name), signal.get_max())
            logger.create_signal_value("{}/Min".format(signal.name), signal.get_min())

        # dump
        if self.current_episode % self.tp.visualization.dump_signals_to_csv_every_x_episodes == 0 \
                and self.current_episode > 0:
            logger.dump_output_csv()

    def reset_game(self, do_not_reset_env=False):
        """
        Resets all the episodic parameters and start a new environment episode.
        :param do_not_reset_env: A boolean that allows prevention of environment reset
        :return: None
        """

        for signal in self.signals:
            signal.reset()
        self.total_reward_in_current_episode = 0
        self.curr_state = {}
        self.last_episode_images = []
        self.current_episode_steps_counter = 0
        self.episode_running_info = {}
        if not do_not_reset_env:
            self.env.reset()
        self.exploration_policy.reset()

        # required for online plotting
        if self.tp.visualization.plot_action_values_online:
            if hasattr(self, 'episode_running_info') and hasattr(self.env, 'actions_description'):
                for action in self.env.actions_description:
                    self.episode_running_info[action] = []
            plt.clf()

        if self.tp.agent.middleware_type == MiddlewareTypes.LSTM:
            for network in self.networks:
                network.online_network.curr_rnn_c_in = network.online_network.middleware_embedder.c_init
                network.online_network.curr_rnn_h_in = network.online_network.middleware_embedder.h_init

        self.prepare_initial_state()

    def preprocess_observation(self, observation):
        """
        Preprocesses the given observation.
        For images - convert to grayscale, resize and convert to int.
        For measurements vectors - normalize by a running average and std.
        :param observation: The agents observation
        :return: A processed version of the observation
        """

        if self.env.is_state_type_image:
            # rescale
            observation = scipy.misc.imresize(observation,
                                              (self.tp.env.desired_observation_height,
                                               self.tp.env.desired_observation_width),
                                              interp=self.tp.rescaling_interpolation_type)
            # rgb to y
            if len(observation.shape) > 2 and observation.shape[2] > 1:
                r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
                observation = 0.2989 * r + 0.5870 * g + 0.1140 * b

            # Render the processed observation which is how the agent will see it
            # Warning: this cannot currently be done in parallel to rendering the environment
            if self.tp.visualization.render_observation:
                if not self.renderer.is_open:
                    self.renderer.create_screen(observation.shape[0], observation.shape[1])
                self.renderer.render_image(observation)

            return observation.astype('uint8')
        else:
            if self.tp.env.normalize_observation and self.sess is not None:
                # standardize the input observation using a running mean and std
                if not self.tp.distributed or not self.tp.agent.share_statistics_between_workers:
                    self.running_observation_stats.push(observation)
                observation = (observation - self.running_observation_stats.mean) / \
                              (self.running_observation_stats.std + 1e-15)
                observation = np.clip(observation, -5.0, 5.0)
            return observation

    def learn_from_batch(self, batch):
        """
        Given a batch of transitions, calculates their target values and updates the network.
        :param batch: A list of transitions
        :return: The loss of the training
        """
        pass

    def train(self):
        """
        A single training iteration. Sample a batch, train on it and update target networks.
        :return: The training loss.
        """
        batch = self.memory.sample(self.tp.batch_size)
        loss = self.learn_from_batch(batch)

        if self.tp.learning_rate_decay_rate != 0:
            self.curr_learning_rate.add_sample(self.tp.sess.run(self.tp.learning_rate))
        else:
            self.curr_learning_rate.add_sample(self.tp.learning_rate)

        # update the target network of every network that has a target network
        if self.total_steps_counter % self.tp.agent.num_steps_between_copying_online_weights_to_target == 0:
            for network in self.networks:
                network.update_target_network(self.tp.agent.rate_for_copying_weights_to_target)
            logger.create_signal_value('Update Target Network', 1)
        else:
            logger.create_signal_value('Update Target Network', 0, overwrite=False)

        return loss

    def extract_batch(self, batch):
        """
        Extracts a single numpy array for each object in a batch of transitions (state, action, etc.)
        :param batch: An array of transitions
        :return: For each transition element, returns a numpy array of all the transitions in the batch
        """
        current_states = {}
        next_states = {}
        current_states['observation'] = np.array([np.array(transition.state['observation']) for transition in batch])
        next_states['observation'] = np.array([np.array(transition.next_state['observation']) for transition in batch])
        actions = np.array([transition.action for transition in batch])
        rewards = np.array([transition.reward for transition in batch])
        game_overs = np.array([transition.game_over for transition in batch])
        total_return = np.array([transition.total_return for transition in batch])

        # get the entire state including measurements if available
        if self.tp.agent.use_measurements:
            current_states['measurements'] = np.array([transition.state['measurements'] for transition in batch])
            next_states['measurements'] = np.array([transition.next_state['measurements'] for transition in batch])

        return current_states, next_states, actions, rewards, game_overs, total_return

    def plot_action_values_online(self):
        """
        Plot an animated graph of the value of each possible action during the episode
        :return: None
        """

        plt.clf()
        for key, data_list in self.episode_running_info.items():
            plt.plot(data_list, label=key)
        plt.legend()
        plt.pause(0.00000001)

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        """
        choose an action to act with in the current episode being played. Different behavior might be exhibited when training
         or testing.

        :param curr_state: the current state to act upon.
        :param phase: the current phase: training or testing.
        :return: chosen action, some action value describing the action (q-value, probability, etc)
        """
        pass

    def preprocess_reward(self, reward):
        if self.tp.env.reward_scaling:
            reward /= float(self.tp.env.reward_scaling)
        if self.tp.env.reward_clipping_max:
            reward = min(reward, self.tp.env.reward_clipping_max)
        if self.tp.env.reward_clipping_min:
            reward = max(reward, self.tp.env.reward_clipping_min)
        return reward

    def tf_input_state(self, curr_state):
        """
        convert curr_state into input tensors tensorflow is expecting.
        """
        # add batch axis with length 1 onto each value
        # extract values from the state based on agent.input_types
        input_state = {}
        for input_name in self.tp.agent.input_types.keys():
            input_state[input_name] = np.expand_dims(np.array(curr_state[input_name]), 0)
        return input_state
        
    def prepare_initial_state(self):
        """
        Create an initial state when starting a new episode
        :return: None
        """
        observation = self.preprocess_observation(self.env.state['observation'])
        self.curr_stack = deque([observation]*self.tp.env.observation_stack_size, maxlen=self.tp.env.observation_stack_size)
        observation = LazyStack(self.curr_stack, -1)

        self.curr_state = {
            'observation': observation
        }
        if self.tp.agent.use_measurements:
            if 'measurements' in self.curr_state.keys():
                self.curr_state['measurements'] = self.env.state['measurements']
            else:
                self.curr_state['measurements'] = np.zeros(0)
            if self.tp.agent.use_accumulated_reward_as_measurement:
                self.curr_state['measurements'] = np.append(self.curr_state['measurements'], 0)

    def act(self, phase=RunPhase.TRAIN):
        """
        Take one step in the environment according to the network prediction and store the transition in memory
        :param phase: Either Train or Test to specify if greedy actions should be used and if transitions should be stored
        :return: A boolean value that signals an episode termination
        """

        if phase != RunPhase.TEST:
            self.total_steps_counter += 1
        self.current_episode_steps_counter += 1

        # get new action
        action_info = {"action_probability": 1.0 / self.env.action_space_size, "action_value": 0, "max_action_value": 0}

        if phase == RunPhase.HEATUP and not self.tp.heatup_using_network_decisions:
            action = self.env.get_random_action()
        else:
            action, action_info = self.choose_action(self.curr_state, phase=phase)

        # perform action
        if type(action) == np.ndarray:
            action = action.squeeze()
        result = self.env.step(action)

        shaped_reward = self.preprocess_reward(result['reward'])
        if 'action_intrinsic_reward' in action_info.keys():
            shaped_reward += action_info['action_intrinsic_reward']
        # TODO: should total_reward_in_current_episode include shaped_reward?
        self.total_reward_in_current_episode += result['reward']
        next_state = copy.copy(result['state'])
        next_state['observation'] = self.preprocess_observation(next_state['observation'])

        # plot action values online
        if self.tp.visualization.plot_action_values_online and phase != RunPhase.HEATUP:
            self.plot_action_values_online()

        # initialize the next state
        # TODO: provide option to stack more than just the observation
        self.curr_stack.append(next_state['observation'])
        observation = LazyStack(self.curr_stack, -1)

        next_state['observation'] = observation
        if self.tp.agent.use_measurements:
            if 'measurements' in result['state'].keys():
                next_state['measurements'] = result['state']['measurements']
            else:
                next_state['measurements'] = np.zeros(0)
            if self.tp.agent.use_accumulated_reward_as_measurement:
                next_state['measurements'] = np.append(next_state['measurements'], self.total_reward_in_current_episode)

        # store the transition only if we are training
        if phase == RunPhase.TRAIN or phase == RunPhase.HEATUP:
            transition = Transition(self.curr_state, result['action'], shaped_reward, next_state, result['done'])
            for key in action_info.keys():
                transition.info[key] = action_info[key]
            if self.tp.agent.add_a_normalized_timestep_to_the_observation:
                transition.info['timestep'] = float(self.current_episode_steps_counter) / self.env.timestep_limit
            self.memory.store(transition)
        elif phase == RunPhase.TEST and self.tp.visualization.dump_gifs:
            # we store the transitions only for saving gifs
            self.last_episode_images.append(self.env.get_rendered_image())

        # update the current state for the next step
        self.curr_state = next_state

        # deal with episode termination
        if result['done']:
            if self.tp.visualization.dump_csv:
                self.update_log(phase=phase)
            self.log_to_screen(phase=phase)

            if phase == RunPhase.TRAIN or phase == RunPhase.HEATUP:
                self.reset_game()

            self.current_episode += 1
            self.tp.current_episode = self.current_episode

        # return episode really ended
        return result['done']

    def evaluate(self, num_episodes, keep_networks_synced=False):
        """
        Run in an evaluation mode for several episodes. Actions will be chosen greedily.
        :param keep_networks_synced: keep the online network in sync with the global network after every episode
        :param num_episodes: The number of episodes to evaluate on
        :return: None
        """

        max_reward_achieved = -float('inf')
        average_evaluation_reward = 0
        screen.log_title("Running evaluation")
        self.env.change_phase(RunPhase.TEST)
        for i in range(num_episodes):
            # keep the online network in sync with the global network
            if keep_networks_synced:
                for network in self.networks:
                    network.sync()

            episode_ended = False
            while not episode_ended:
                episode_ended = self.act(phase=RunPhase.TEST)

                if keep_networks_synced \
                   and self.total_steps_counter % self.tp.agent.update_evaluation_agent_network_after_every_num_steps:
                    for network in self.networks:
                        network.sync()

            if self.total_reward_in_current_episode > max_reward_achieved:
                max_reward_achieved = self.total_reward_in_current_episode
                frame_skipping = int(5/self.tp.env.frame_skip)
                if self.tp.visualization.dump_gifs:
                    logger.create_gif(self.last_episode_images[::frame_skipping],
                                      name='score-{}'.format(max_reward_achieved), fps=10)

            average_evaluation_reward += self.total_reward_in_current_episode
            self.reset_game()

        average_evaluation_reward /= float(num_episodes)

        self.env.change_phase(RunPhase.TRAIN)
        screen.log_title("Evaluation done. Average reward = {}.".format(average_evaluation_reward))

    def post_training_commands(self):
        pass

    def improve(self):
        """
        Training algorithms wrapper. Heatup >> [ Evaluate >> Play >> Train >> Save checkpoint ]

        :return: None
        """

        # synchronize the online network weights with the global network
        for network in self.networks:
            network.sync()

        # heatup phase
        if self.tp.num_heatup_steps != 0:
            self.in_heatup = True
            screen.log_title("Starting heatup {}".format(self.task_id))
            num_steps_required_for_one_training_batch = self.tp.batch_size * self.tp.env.observation_stack_size
            for step in range(max(self.tp.num_heatup_steps, num_steps_required_for_one_training_batch)):
                self.act(phase=RunPhase.HEATUP)

        # training phase
        self.in_heatup = False
        screen.log_title("Starting training {}".format(self.task_id))
        self.exploration_policy.change_phase(RunPhase.TRAIN)
        training_start_time = time.time()
        model_snapshots_periods_passed = -1
        self.reset_game()

        while self.training_iteration < self.tp.num_training_iterations:
            # evaluate
            evaluate_agent = (self.last_episode_evaluation_ran is not self.current_episode) and \
                             (self.current_episode % self.tp.evaluate_every_x_episodes == 0)
            evaluate_agent = evaluate_agent or \
                             (self.imitation and self.training_iteration > 0 and
                              self.training_iteration % self.tp.evaluate_every_x_training_iterations == 0)

            if evaluate_agent:
                self.env.reset(force_environment_reset=True)
                self.last_episode_evaluation_ran = self.current_episode
                self.evaluate(self.tp.evaluation_episodes)

            # snapshot model
            if self.tp.save_model_sec and self.tp.save_model_sec > 0 and not self.tp.distributed:
                total_training_time = time.time() - training_start_time
                current_snapshot_period = (int(total_training_time) // self.tp.save_model_sec)
                if current_snapshot_period > model_snapshots_periods_passed:
                    model_snapshots_periods_passed = current_snapshot_period
                    self.save_model(model_snapshots_periods_passed)
                    if hasattr(self, 'running_observation_state') and self.running_observation_stats is not None:
                        to_pickle(self.running_observation_stats,
                                  os.path.join(self.tp.save_model_dir,
                                               "running_stats.p".format(model_snapshots_periods_passed)))

            # play and record in replay buffer
            if self.tp.agent.collect_new_data:
                if self.tp.agent.step_until_collecting_full_episodes:
                    step = 0
                    while step < self.tp.agent.num_consecutive_playing_steps or self.memory.get_episode(-1).length() != 0:
                        self.act()
                        step += 1
                else:
                    for step in range(self.tp.agent.num_consecutive_playing_steps):
                        self.act()

            # train
            if self.tp.train:
                for step in range(self.tp.agent.num_consecutive_training_steps):
                    loss = self.train()
                    self.loss.add_sample(loss)
                    self.training_iteration += 1
                    if self.imitation:
                        self.log_to_screen(RunPhase.TRAIN)
                self.post_training_commands()

    def save_model(self, model_id):
        self.main_network.save_model(model_id)
