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
from copy import deepcopy
from typing import Tuple, List, Union

from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.nec_agent import NECAgentParameters
from rl_coach.base_parameters import AgentParameters, VisualizationParameters, TaskParameters, \
    PresetValidationParameters
from rl_coach.core_types import RunPhase
from rl_coach.environments.environment import EnvironmentParameters, Environment
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

from rl_coach.level_manager import LevelManager
from rl_coach.logger import screen
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import short_dynamic_import

from rl_coach.memories.episodic import EpisodicExperienceReplayParameters

from rl_coach.core_types import TimeTypes


class BatchRLGraphManager(BasicRLGraphManager):
    """
    A batch RL graph manager creates scenario of learning from a dataset without a simulator.
    """
    def __init__(self, agent_params: AgentParameters, env_params: Union[EnvironmentParameters, None],
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters = VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters(),
                 name='batch_rl_graph', spaces_definition: SpacesDefinition = None, reward_model_num_epochs: int = 100,
                 train_to_eval_ratio: float = 0.8):

        super().__init__(agent_params, env_params, schedule_params, vis_params, preset_validation_params, name)
        self.is_batch_rl = True
        self.time_metric = TimeTypes.Epoch
        self.reward_model_num_epochs = reward_model_num_epochs
        self.spaces_definition = spaces_definition

        # setting this here to make sure that, by default, train_to_eval_ratio gets a value < 1
        # (its default value in the memory is 1)
        self.agent_params.memory.train_to_eval_ratio = train_to_eval_ratio

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        if self.env_params:
            # environment loading
            self.env_params.seed = task_parameters.seed
            self.env_params.experiment_path = task_parameters.experiment_path
            env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                             visualization_parameters=self.visualization_parameters)
        else:
            env = None

        # Only DQN variants and NEC are supported at this point.
        assert(isinstance(self.agent_params, DQNAgentParameters) or isinstance(self.agent_params, NECAgentParameters))
        # Only Episodic memories are supported,
        # for evaluating the sequential doubly robust estimator
        assert(isinstance(self.agent_params.memory, EpisodicExperienceReplayParameters))

        # agent loading
        self.agent_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.agent_params.name = "agent"
        self.agent_params.is_batch_rl_training = True

        if 'reward_model' not in self.agent_params.network_wrappers:
            # user hasn't defined params for the reward model. we will use the same params as used for the 'main'
            # network.
            self.agent_params.network_wrappers['reward_model'] = deepcopy(self.agent_params.network_wrappers['main'])

        agent = short_dynamic_import(self.agent_params.path)(self.agent_params)

        if not env and not self.agent_params.memory.load_memory_from_file_path:
            screen.warning("A BatchRLGraph requires setting a dataset to load into the agent's memory or alternatively "
                           "using an environment to create a (random) dataset from. This agent should only be used for "
                           "inference. ")
        # set level manager
        level_manager = LevelManager(agents=agent, environment=env, name="main_level",
                                     spaces_definition=self.spaces_definition)

        if env:
            return [level_manager], [env]
        else:
            return [level_manager], []

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

        # initialize the network parameters from the global network
        self.sync()

        # TODO a bug in heatup where the last episode run is not fed into the ER. e.g. asked for 1024 heatup steps,
        #  last ran episode ended increased the total to 1040 steps, but the ER will contain only 1014 steps.
        #  The last episode is not there. Is this a bug in my changes or also on master?

        # Creating a dataset during the heatup phase is useful mainly for tutorial and debug purposes. If we have both
        # an environment and a dataset to load from, we will use the environment only for evaluating the policy,
        # and will not run heatup.

        # heatup
        if self.env_params is not None and not self.agent_params.memory.load_memory_from_file_path:
            self.heatup(self.heatup_steps)

        # from this point onwards, the dataset cannot be changed anymore. Allows for performance improvements.
        self.level_managers[0].agents['agent'].memory.freeze()

        self.initialize_ope_models_and_stats()

        # improve
        if self.task_parameters.task_index is not None:
            screen.log_title("Starting to improve {} task index {}".format(self.name, self.task_parameters.task_index))
        else:
            screen.log_title("Starting to improve {}".format(self.name))

        # the outer most training loop
        improve_steps_end = self.total_steps_counters[RunPhase.TRAIN] + self.improve_steps
        while self.total_steps_counters[RunPhase.TRAIN] < improve_steps_end:
            # TODO if we have an environment, do we want to use it to have the agent train against, and use the
            #  collected replay buffer as a dataset? (as oppose to what we currently have, where the dataset is built
            #  during heatup, and is composed on random actions)
            # perform several steps of training
            if self.steps_between_evaluation_periods.num_steps > 0:
                with self.phase_context(RunPhase.TRAIN):
                    self.reset_internal_state(force_environment_reset=True)

                    steps_between_evaluation_periods_end = self.current_step_counter + self.steps_between_evaluation_periods
                    while self.current_step_counter < steps_between_evaluation_periods_end:
                        self.train()

            # the output of batch RL training is always a checkpoint of the trained agent. we always save a checkpoint,
            # each epoch, regardless of the user's command line arguments.
            self.save_checkpoint()

            # run off-policy evaluation estimators to evaluate the agent's performance against the dataset
            self.run_off_policy_evaluation()

            if self.env_params is not None and self.evaluate(self.evaluation_steps):
                # if we do have a simulator (although we are in a batch RL setting we might have a simulator, e.g. when
                # demonstrating the batch RL use-case using one of the existing Coach environments),
                # we might want to evaluate vs. the simulator every now and then.
                break

    def initialize_ope_models_and_stats(self):
        """

        :return:
        """
        agent = self.level_managers[0].agents['agent']

        screen.log_title("Training a regression model for estimating MDP rewards")
        agent.improve_reward_model(epochs=self.reward_model_num_epochs)

        # prepare dataset to be consumed in the expected formats for OPE
        agent.memory.prepare_evaluation_dataset()

        screen.log_title("Collecting static statistics for OPE")
        agent.ope_manager.gather_static_shared_stats(evaluation_dataset_as_transitions=
                                                     agent.memory.evaluation_dataset_as_transitions,
                                                     batch_size=agent.ap.network_wrappers['main'].batch_size,
                                                     reward_model=agent.networks['reward_model'].online_network,
                                                     network_keys=list(agent.ap.network_wrappers['main'].
                                                                       input_embedders_parameters.keys()))

    def run_off_policy_evaluation(self):
        """
        Run off-policy evaluation estimators to evaluate the trained policy performance against the dataset
        :return:
        """
        self.level_managers[0].agents['agent'].run_off_policy_evaluation()



