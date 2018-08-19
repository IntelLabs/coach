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
from rl_coach.environments.environment import EnvironmentParameters, Environment
from rl_coach.graph_managers.graph_manager import GraphManager, ScheduleParameters
from rl_coach.level_manager import LevelManager
from rl_coach.utils import short_dynamic_import


class BasicRLGraphManager(GraphManager):
    """
    A basic RL graph manager creates the common scheme of RL where there is a single agent which interacts with a
    single environment.
    """
    def __init__(self, agent_params: AgentParameters, env_params: EnvironmentParameters,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters=VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters()):
        super().__init__('simple_rl_graph', schedule_params, vis_params)
        self.agent_params = agent_params
        self.env_params = env_params
        self.preset_validation_params = preset_validation_params

        self.agent_params.visualization = vis_params
        if self.agent_params.input_filter is None:
            self.agent_params.input_filter = env_params.default_input_filter()
        if self.agent_params.output_filter is None:
            self.agent_params.output_filter = env_params.default_output_filter()

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        # environment loading
        self.env_params.seed = task_parameters.seed
        self.env_params.experiment_path = task_parameters.experiment_path
        env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                         visualization_parameters=self.visualization_parameters)

        # agent loading
        self.agent_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.agent_params.name = "agent"
        agent = short_dynamic_import(self.agent_params.path)(self.agent_params)

        # set level manager
        level_manager = LevelManager(agents=agent, environment=env, name="main_level")

        return [level_manager], [env]
