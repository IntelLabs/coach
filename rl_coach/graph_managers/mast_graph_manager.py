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
import sys

from rl_coach.base_parameters import AgentParameters, VisualizationParameters, \
    PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, RunPhase, TrainingSteps
from rl_coach.data_stores.redis_data_store import RedisDataStore
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.logger import screen, Logger


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
        self.last_publish_time = 0

    def actor(self, total_steps_to_act: EnvironmentSteps, data_store: RedisDataStore):
        self.verify_graph_was_created()

        # initialize the network parameters from the global network
        self.sync()

        # act
        screen.log_title("{}-actor{}: Starting to act on the environment".format(self.name, self.task_parameters.task_index))

        # perform several steps of training interleaved with acting
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
                        data_store.attempt_load_policy(self)
                    self.act(EnvironmentSteps(1))

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
                    self.train()
                    if (self.current_step_counter[EnvironmentSteps] - self.last_publish_time) > 90000:  # TODO extract hyper-param
                        print("publishing")
                        self.last_publish_time = self.current_step_counter[EnvironmentSteps]
                        data_store.save_policy(self)
                        self.occasionally_save_checkpoint()

    def fetch_from_worker(self, num_consecutive_playing_steps=None):
        if hasattr(self, 'memory_backend'):
            with self.phase_context(RunPhase.TRAIN):
                # for transition in self.memory_backend.fetch_subscribe_all_msgs(num_consecutive_playing_steps):
                for transition in self.memory_backend.fetch(num_consecutive_playing_steps):
                    self.emulate_act_on_trainer(EnvironmentSteps(1), transition)


# TODO POMODORO trainer cannot keep up with the actors input rate to the queue, thus becoming very off policy. how come? also need to fix the shared running stats to also be published every 2048 steps