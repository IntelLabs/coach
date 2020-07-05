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
import os
import sys
from collections import OrderedDict

from rl_coach.base_parameters import AgentParameters, VisualizationParameters, \
    PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, RunPhase, TrainingSteps, EnvironmentEpisodes
from rl_coach.data_stores.redis_data_store import RedisDataStore
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.logger import screen


class MASTGraphManager(BasicRLGraphManager):
    """
    A basic RL graph manager creates the common scheme of RL where there is a single agent which interacts with a
    single environment.
    """
    def __init__(self, agent_params: AgentParameters, env_params: EnvironmentParameters,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters=VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters(),
                 name='mast_rl_graph'):
        super().__init__(agent_params=agent_params, env_params=env_params, name=name, schedule_params=schedule_params,
                         vis_params=vis_params, preset_validation_params=preset_validation_params)
        self.first_policy_publish = False
        self.last_publish_step = 0
        self.latest_published_policy_id = 0

    def actor(self, total_steps_to_act: EnvironmentSteps, data_store: RedisDataStore):
        self.verify_graph_was_created()

        # initialize the network parameters from the global network
        self.sync()

        # act
        screen.log_title("{}-actor{}: Starting to act on the environment".format(self.name, self.task_parameters.task_index))

        if total_steps_to_act.num_steps > 0:
            with self.phase_context(RunPhase.TRAIN):
                self.reset_internal_state(force_environment_reset=True)

                count_end = self.current_step_counter + total_steps_to_act
                while self.current_step_counter < count_end:
                    # The actual number of steps being done on the environment
                    # is decided by the agent, though this inner loop always
                    # takes at least one step in the environment (at the GraphManager level).
                    # The agent might also decide to skip acting altogether.
                    # Depending on internal counters and parameters, it doesn't always train or save checkpoints.
                    if data_store.end_of_policies():
                        break
                    if self.current_step_counter[EnvironmentSteps] % 100 == 0:  # TODO extract hyper-param
                        if data_store.attempt_load_policy(self):
                            log = OrderedDict()
                            log['Loading new policy'] = self.latest_published_policy_id
                            screen.log_dict(log, prefix='Actor {}'.format(
                                self.task_parameters.task_index))

                    self.act(EnvironmentSteps(1))

    def evaluate(self, total_steps_to_act: EnvironmentSteps, data_store: RedisDataStore) -> bool:
        self.verify_graph_was_created()

        # initialize the network parameters from the global network
        self.sync()

        # act
        screen.log_title("{}-evaluator: Starting to test the policy on the environment".format(self.name))

        if total_steps_to_act.num_steps > 0:
            with self.phase_context(RunPhase.TEST):
                self.reset_internal_state(force_environment_reset=True)

                count_end = self.current_step_counter + total_steps_to_act
                while self.current_step_counter < count_end:
                    # The actual number of steps being done on the environment
                    # is decided by the agent, though this inner loop always
                    # takes at least one step in the environment (at the GraphManager level).
                    # The agent might also decide to skip acting altogether.
                    # Depending on internal counters and parameters, it doesn't always train or save checkpoints.
                    if data_store.end_of_policies():
                        break

                    if data_store.attempt_load_policy(self):
                        log = OrderedDict()
                        log['Loading new policy'] = ""
                        screen.log_dict(log, prefix='Evaluator')

                    # In case of an evaluation-only worker, fake a phase transition before and after every
                    # episode to make sure results are logged correctly
                    if self.task_parameters.evaluate_only is not None:
                        self.phase = RunPhase.TEST
                    self.act(EnvironmentEpisodes(1))
                    self.sync()
                    if self.task_parameters.evaluate_only is not None:
                        self.phase = RunPhase.TRAIN

        if self.should_stop():
            self.flush_finished()
            screen.success("Reached required success rate. Exiting.")
            return True
        return False

    def trainer(self, total_steps_to_train: TrainingSteps, data_store: RedisDataStore):
        self.verify_graph_was_created()

        # initialize the network parameters from the global network
        self.sync()

        self.setup_memory_backend()

        # train
        screen.log_title("{}-trainer{}: Starting to train from collected experience".format(self.name, self.task_parameters.task_index))

        # perform several steps of training interleaved with acting
        if total_steps_to_train.num_steps > 0:
            with self.phase_context(RunPhase.TRAIN):
                self.reset_internal_state(force_environment_reset=True)

                count_end = self.current_step_counter + total_steps_to_train
                while self.current_step_counter < count_end:
                    # The actual number of steps being done on the environment
                    # is decided by the agent, though this inner loop always
                    # takes at least one step in the environment (at the GraphManager level).
                    # The agent might also decide to skip acting altogether.
                    # Depending on internal counters and parameters, it doesn't always train or save checkpoints.
                    self.fetch_from_worker(self.agent_params.algorithm.num_consecutive_playing_steps)
                    # if (self.get_agent().total_steps_counter - self.last_publish_step) >= 20000:
                    #     list(list(self.get_agent().pre_network_filter._observation_filters.values())[0].values())[
                    #         0].running_observation_stats.publish_on_next_push = True
                    self.train()
                    if (self.get_agent().total_steps_counter - self.last_publish_step) >= 20000:  # TODO extract hyper-param
                        data_store.save_policy(self)
                        self.occasionally_save_checkpoint()

                        log = OrderedDict()
                        log['Publishing a new policy'] = self.latest_published_policy_id
                        screen.log_dict(log, prefix='Trainer')

                        self.latest_published_policy_id += 1
                        self.last_publish_step = self.current_step_counter[EnvironmentSteps]

    def fetch_from_worker(self, num_consecutive_playing_steps=None):
        if hasattr(self, 'memory_backend'):
            with self.phase_context(RunPhase.TRAIN):
                # for transition in self.memory_backend.fetch_subscribe_all_msgs(num_consecutive_playing_steps):
                #     self.emulate_act_on_trainer(EnvironmentSteps(1), transition)


                ##### an alternative flow to balaji's which is more efficient and gets full episodes at a time #####
                ##### - the current issue with it is that since the agen't flow emulation does not take place, #####
                #####   signals are not tracked and are missing from dashboard.                                #####

                for episode in self.memory_backend.fetch_subscribe_all_msgs(num_consecutive_playing_steps):
                    self.get_agent().memory.store_episode(episode)

                    # -  book-keeping -
                    self.get_agent().total_steps_counter += episode.length()
                    self.get_agent().current_episode += 1

                    # last episode is always complete as we're getting full episodes from actors.
                    # this is required so that agent._should_update() will be aware that the last episode is complete
                    self.get_agent().current_episode_buffer.is_complete = True

                    self.current_step_counter[EnvironmentSteps] = self.get_agent().total_steps_counter
                    log = OrderedDict()
                    log['Total steps fetched from actors'] = self.get_agent().total_steps_counter
                    log['Last episode from actor ID'] = episode.task_id
                    log['Episode ID'] = episode.episode_id
                    screen.log_dict(log, prefix='Trainer')

# TODO some bug where when running with an eval agent cartpole starts with a reward of 200. probably loading some old policy which wasn't cleaned from Redis