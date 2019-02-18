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
from typing import Tuple, List

from rl_coach.base_parameters import AgentParameters, VisualizationParameters, TaskParameters, \
    PresetValidationParameters
from rl_coach.core_types import RunPhase
from rl_coach.environments.environment import EnvironmentParameters, Environment
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

from rl_coach.level_manager import LevelManager
from rl_coach.logger import screen
from rl_coach.utils import short_dynamic_import


class BatchRLGraphManager(BasicRLGraphManager):
    """
    A batch RL graph manager creates scenario of learning from a dataset without a simulator.
    """
    def __init__(self, agent_params: AgentParameters, env_params: EnvironmentParameters,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters = VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters(),
                 name='batch_rl_graph'):

        super().__init__(agent_params, env_params, schedule_params, vis_params, preset_validation_params, name)
        self.is_batch_rl = True

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        if self.env_params:
            # environment loading
            self.env_params.seed = task_parameters.seed
            self.env_params.experiment_path = task_parameters.experiment_path
            env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                             visualization_parameters=self.visualization_parameters)
        else:
            env = None

        # agent loading
        self.agent_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.agent_params.name = "agent"
        self.agent_params.is_batch_rl_training = True
        agent = short_dynamic_import(self.agent_params.path)(self.agent_params)

        # TODO load dataset into the agent's replay buffer. optionally normalize it, and clean it from outliers.
        # agent.memory.load_dataset()

        # set level manager
        # TODO how to handle not having an environment?
        #  allow for no environment through the entire pipe?
        #  use a dummy batch RL env?
        level_manager = LevelManager(agents=agent, environment=env, name="main_level")

        return [level_manager], [env]

    def improve(self):
        """
        The main loop of the run.
        Defined in the following steps:
        1. Heatup
        2. Repeat:
            2.1. Repeat:
                2.1.1. Train
                2.1.2. Possibly save checkpoint
            2.2. Evaluate
        :return: None
        """

        self.verify_graph_was_created()

        # TODO handle unsupported use-cases for batch RL. i.e. on-policy algorithms,
        #  distributed setting (either local or multi-node)

        # initialize the network parameters from the global network
        self.sync()

        # heatup
        if self.env_params is not None:
            self.heatup(self.heatup_steps)

        # improve
        if self.task_parameters.task_index is not None:
            screen.log_title("Starting to improve {} task index {}".format(self.name, self.task_parameters.task_index))
        else:
            screen.log_title("Starting to improve {}".format(self.name))

        # the outer most training loop
        improve_steps_end = self.total_steps_counters[RunPhase.TRAIN] + self.improve_steps
        while self.total_steps_counters[RunPhase.TRAIN] < improve_steps_end:

            # perform several steps of training
            if self.steps_between_evaluation_periods.num_steps > 0:
                with self.phase_context(RunPhase.TRAIN):
                    self.reset_internal_state(force_environment_reset=True)

                    steps_between_evaluation_periods_end = self.current_step_counter + self.steps_between_evaluation_periods
                    while self.current_step_counter < steps_between_evaluation_periods_end:
                        self.train()

            # the output of batch RL training is always a checkpoint of the trained agent. we always save a checkpoint,
            # regardless of the user's command line arguments.
            # self.save_checkpoint()

            # TODO should this be done in the GraphManager level or in the agent's level?
            # run off-policy evaluation estimators to evaluate the agent's performance against the dataset
            self.run_ope()

            if self.env_params is not None and self.evaluate(self.evaluation_steps):
                # if we do have a simulator (although we are in a batch RL setting we might have a simulator, e.g. when
                # demonstrating the batch RL use-case using one of the existing Coach environments),
                # we might want to evaluate vs. the simulator every now and then.
                break

    def run_ope(self):
        """
        Run off-policy evaluation estimators to evaluate the trained policy performance against the dataset
        :return:
        """
        [manager.run_ope() for manager in self.level_managers]

