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
import time
from collections import OrderedDict
from distutils.dir_util import copy_tree, remove_tree
from typing import List, Tuple

from rl_coach.base_parameters import iterable_to_items, TaskParameters, DistributedTaskParameters, \
    VisualizationParameters, \
    Parameters, PresetValidationParameters
from rl_coach.core_types import TotalStepsCounter, RunPhase, PlayingStepsType, TrainingSteps, EnvironmentEpisodes, \
    EnvironmentSteps, \
    StepMethod
from rl_coach.environments.environment import Environment
from rl_coach.level_manager import LevelManager
from rl_coach.logger import screen, Logger
from rl_coach.utils import set_cpu


class ScheduleParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.heatup_steps = None
        self.evaluation_steps = None
        self.steps_between_evaluation_periods = None
        self.improve_steps = None


class HumanPlayScheduleParameters(ScheduleParameters):
    def __init__(self):
        super().__init__()
        self.heatup_steps = EnvironmentSteps(0)
        self.evaluation_steps = EnvironmentEpisodes(0)
        self.steps_between_evaluation_periods = EnvironmentEpisodes(100000000)
        self.improve_steps = TrainingSteps(10000000000)


class GraphManager(object):
    """
    A graph manager is responsible for creating and initializing a graph of agents, including all its internal
    components. It is then used in order to schedule the execution of operations on the graph, such as acting and
    training.
    """
    def __init__(self,
                 name: str,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters = VisualizationParameters()):
        self.sess = None
        self.level_managers = []
        self.top_level_manager = None
        self.environments = []
        self.heatup_steps = schedule_params.heatup_steps
        self.evaluation_steps = schedule_params.evaluation_steps
        self.steps_between_evaluation_periods = schedule_params.steps_between_evaluation_periods
        self.improve_steps = schedule_params.improve_steps
        self.visualization_parameters = vis_params
        self.name = name
        self.task_parameters = None
        self._phase = self.phase = RunPhase.UNDEFINED
        self.preset_validation_params = PresetValidationParameters()
        self.reset_required = False

        # timers
        self.graph_initialization_time = time.time()
        self.heatup_start_time = None
        self.training_start_time = None
        self.last_evaluation_start_time = None
        self.last_checkpoint_saving_time = time.time()

        # counters
        self.total_steps_counters = {
            RunPhase.HEATUP: TotalStepsCounter(),
            RunPhase.TRAIN: TotalStepsCounter(),
            RunPhase.TEST: TotalStepsCounter()
        }
        self.checkpoint_id = 0

        self.checkpoint_saver = None
        self.graph_logger = Logger()

    def create_graph(self, task_parameters: TaskParameters):
        self.task_parameters = task_parameters

        if isinstance(task_parameters, DistributedTaskParameters):
            screen.log_title("Creating graph - name: {} task id: {} type: {}".format(self.__class__.__name__,
                                                                                     task_parameters.task_index,
                                                                                     task_parameters.job_type))
        else:
            screen.log_title("Creating graph - name: {}".format(self.__class__.__name__))

        # "hide" the gpu if necessary
        if task_parameters.use_cpu:
            set_cpu()

        # create a target server for the worker and a device
        if isinstance(task_parameters, DistributedTaskParameters):
            task_parameters.worker_target, task_parameters.device = \
                self.create_worker_or_parameters_server(task_parameters=task_parameters)

        # create the graph modules
        self.level_managers, self.environments = self._create_graph(task_parameters)

        # set self as the parent of all the level managers
        self.top_level_manager = self.level_managers[0]
        for level_manager in self.level_managers:
            level_manager.parent_graph_manager = self

        # create a session (it needs to be created after all the graph ops were created)
        self.sess = None
        self.create_session(task_parameters=task_parameters)

        self._phase = self.phase = RunPhase.UNDEFINED

        self.setup_logger()

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        """
        Create all the graph modules and the graph scheduler
        :param task_parameters: the parameters of the task
        :return: the initialized level managers and environments
        """
        raise NotImplementedError("")

    def create_worker_or_parameters_server(self, task_parameters: DistributedTaskParameters):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
        config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1

        from rl_coach.architectures.tensorflow_components.distributed_tf_utils import create_and_start_parameters_server, \
            create_cluster_spec, create_worker_server_and_device

        # create cluster spec
        cluster_spec = create_cluster_spec(parameters_server=task_parameters.parameters_server_hosts,
                                           workers=task_parameters.worker_hosts)

        # create and start parameters server (non-returning function) or create a worker and a device setter
        if task_parameters.job_type == "ps":
            create_and_start_parameters_server(cluster_spec=cluster_spec,
                                               config=config)
        elif task_parameters.job_type == "worker":
            return create_worker_server_and_device(cluster_spec=cluster_spec,
                                                   task_index=task_parameters.task_index,
                                                   use_cpu=task_parameters.use_cpu,
                                                   config=config)
        else:
            raise ValueError("The job type should be either ps or worker and not {}"
                             .format(task_parameters.job_type))

    def create_session(self, task_parameters: DistributedTaskParameters):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
        config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1

        if isinstance(task_parameters, DistributedTaskParameters):
            # the distributed tensorflow setting
            from rl_coach.architectures.tensorflow_components.distributed_tf_utils import create_monitored_session
            if hasattr(self.task_parameters, 'checkpoint_restore_dir') and self.task_parameters.checkpoint_restore_dir:
                checkpoint_dir = os.path.join(task_parameters.experiment_path, 'checkpoint')
                if os.path.exists(checkpoint_dir):
                    remove_tree(checkpoint_dir)
                copy_tree(task_parameters.checkpoint_restore_dir, checkpoint_dir)
            else:
                checkpoint_dir = task_parameters.save_checkpoint_dir

            self.sess = create_monitored_session(target=task_parameters.worker_target,
                                                 task_index=task_parameters.task_index,
                                                 checkpoint_dir=checkpoint_dir,
                                                 save_checkpoint_secs=task_parameters.save_checkpoint_secs,
                                                 config=config)
            # set the session for all the modules
            self.set_session(self.sess)
        else:
            self.variables_to_restore = tf.global_variables()
            self.variables_to_restore = [v for v in self.variables_to_restore if '/online' in v.name]
            self.checkpoint_saver = tf.train.Saver(self.variables_to_restore)

            # regular session
            self.sess = tf.Session(config=config)

            # set the session for all the modules
            self.set_session(self.sess)

            # restore from checkpoint if given
            self.restore_checkpoint()

    def setup_logger(self) -> None:
        # dump documentation
        logger_prefix = "{graph_name}".format(graph_name=self.name)
        self.graph_logger.set_logger_filenames(self.task_parameters.experiment_path, logger_prefix=logger_prefix,
                                               add_timestamp=True, task_id=self.task_parameters.task_index)
        if self.visualization_parameters.dump_parameters_documentation:
            self.graph_logger.dump_documentation(str(self))
        [manager.setup_logger() for manager in self.level_managers]

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the graph
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the graph and all the hierarchy levels below it
        :param val: the new phase
        :return: None
        """
        self._phase = val
        for level_manager in self.level_managers:
            level_manager.phase = val
        for environment in self.environments:
            environment.phase = val

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the modules in the graph
        :return: None
        """
        [manager.set_session(sess) for manager in self.level_managers]

    def heatup(self, steps: PlayingStepsType) -> None:
        """
        Perform heatup for several steps, which means taking random actions and storing the results in memory
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        steps_copy = copy.copy(steps)

        if steps_copy.num_steps > 0:
            self.phase = RunPhase.HEATUP
            screen.log_title("{}: Starting heatup".format(self.name))
            self.heatup_start_time = time.time()

            # reset all the levels before starting to heatup
            self.reset_internal_state(force_environment_reset=True)

            # act on the environment
            while steps_copy.num_steps > 0:
                steps_done, _ = self.act(steps_copy, continue_until_game_over=True, return_on_game_over=True)
                steps_copy.num_steps -= steps_done

            # training phase
            self.phase = RunPhase.UNDEFINED

    def handle_episode_ended(self) -> None:
        """
        End an episode and reset all the episodic parameters
        :return: None
        """
        self.total_steps_counters[self.phase][EnvironmentEpisodes] += 1

        # TODO: we should disentangle ending the episode from resetting the internal state
        # self.reset_internal_state()

    def train(self, steps: TrainingSteps) -> None:
        """
        Perform several training iterations for all the levels in the hierarchy
        :param steps: number of training iterations to perform
        :return: None
        """
        # perform several steps of training interleaved with acting
        count_end = self.total_steps_counters[RunPhase.TRAIN][TrainingSteps] + steps.num_steps
        while self.total_steps_counters[RunPhase.TRAIN][TrainingSteps] < count_end:
            self.total_steps_counters[RunPhase.TRAIN][TrainingSteps] += 1
            [manager.train() for manager in self.level_managers]

    def reset_internal_state(self, force_environment_reset=False) -> None:
        """
        Reset an episode for all the levels
        :param force_environment_reset: force the environment to reset the episode even if it has some conditions that
                                        tell it not to. for example, if ale life is lost, gym will tell the agent that
                                        the episode is finished but won't actually reset the episode if there are more
                                        lives available
        :return: None
        """
        self.reset_required = False
        [environment.reset_internal_state(force_environment_reset) for environment in self.environments]
        [manager.reset_internal_state() for manager in self.level_managers]

    def act(self, steps: PlayingStepsType, return_on_game_over: bool=False, continue_until_game_over=False,
            keep_networks_in_sync=False) -> (int, bool):
        """
        Do several steps of acting on the environment
        :param steps: the number of steps as a tuple of steps time and steps count
        :param return_on_game_over: finish acting if an episode is finished
        :param continue_until_game_over: continue playing until an episode was completed
        :param keep_networks_in_sync: sync the network parameters with the global network before each episode
        :return: the actual number of steps done, a boolean value that represent if the episode was done when finishing
                 the function call
        """
        # perform several steps of playing
        result = None

        hold_until_a_full_episode = True if continue_until_game_over else False
        initial_count = self.total_steps_counters[self.phase][steps.__class__]
        count_end = initial_count + steps.num_steps

        # The assumption here is that the total_steps_counters are each updated when an event
        #  takes place (i.e. an episode ends)
        # TODO - The counter of frames is not updated correctly. need to fix that.
        while self.total_steps_counters[self.phase][steps.__class__] < count_end or hold_until_a_full_episode:
            # reset the environment if the previous episode was terminated
            if self.reset_required:
                self.reset_internal_state()

            current_steps = self.environments[0].total_steps_counter

            result = self.top_level_manager.step(None)
            # result will be None if at least one level_manager decided not to play (= all of his agents did not play)
            # causing the rest of the level_managers down the stack not to play either, and thus the entire graph did
            # not act
            if result is None:
                break

            # add the diff between the total steps before and after stepping, such that environment initialization steps
            # (like in Atari) will not be counted.
            # We add at least one step so that even if no steps were made (in case no actions are taken in the training
            # phase), the loop will end eventually.
            self.total_steps_counters[self.phase][EnvironmentSteps] += \
                max(1, self.environments[0].total_steps_counter - current_steps)

            if result.game_over:
                hold_until_a_full_episode = False
                self.handle_episode_ended()
                self.reset_required = True
                if keep_networks_in_sync:
                    self.sync_graph()
                if return_on_game_over:
                    return self.total_steps_counters[self.phase][EnvironmentSteps] - initial_count, True

        # return the game over status
        if result:
            return self.total_steps_counters[self.phase][EnvironmentSteps] - initial_count, result.game_over
        else:
            return self.total_steps_counters[self.phase][EnvironmentSteps] - initial_count, False

    def train_and_act(self, steps: StepMethod) -> None:
        """
        Train the agent by doing several acting steps followed by several training steps continually
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        # perform several steps of training interleaved with acting
        if steps.num_steps > 0:
            self.phase = RunPhase.TRAIN
            count_end = self.total_steps_counters[self.phase][steps.__class__] + steps.num_steps
            self.reset_internal_state(force_environment_reset=True)
            #TODO - the below while loop should end with full episodes, so to avoid situations where we have partial
            #  episodes in memory
            while self.total_steps_counters[self.phase][steps.__class__] < count_end:
                # The actual steps being done on the environment are decided by the agents themselves.
                # This is just an high-level controller.
                self.act(EnvironmentSteps(1))
                self.train(TrainingSteps(1))
                self.save_checkpoint()
            self.phase = RunPhase.UNDEFINED

    def sync_graph(self) -> None:
        """
        Sync the global network parameters to the graph
        :return:
        """
        [manager.sync() for manager in self.level_managers]

    def evaluate(self, steps: PlayingStepsType, keep_networks_in_sync: bool=False) -> None:
        """
        Perform evaluation for several steps
        :param steps: the number of steps as a tuple of steps time and steps count
        :param keep_networks_in_sync: sync the network parameters with the global network before each episode
        :return: None
        """
        if steps.num_steps > 0:
            self.phase = RunPhase.TEST
            self.last_evaluation_start_time = time.time()

            # reset all the levels before starting to evaluate
            self.reset_internal_state(force_environment_reset=True)
            self.sync_graph()

            count_end = self.total_steps_counters[self.phase][steps.__class__] + steps.num_steps
            while self.total_steps_counters[self.phase][steps.__class__] < count_end:
                steps_done, _ = self.act(steps, continue_until_game_over=True, return_on_game_over=True,
                                         keep_networks_in_sync=keep_networks_in_sync)

            self.phase = RunPhase.UNDEFINED

    def restore_checkpoint(self):
        # TODO: find better way to load checkpoints that were saved with a global network into the online network
        if hasattr(self.task_parameters, 'checkpoint_restore_dir') and self.task_parameters.checkpoint_restore_dir:
            import tensorflow as tf
            checkpoint_dir = self.task_parameters.checkpoint_restore_dir
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            screen.log_title("Loading checkpoint: {}".format(checkpoint.model_checkpoint_path))
            variables = {}
            for var_name, _ in tf.contrib.framework.list_variables(self.task_parameters.checkpoint_restore_dir):
                # Load the variable
                var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

                # Set the new name
                new_name = var_name
                new_name = new_name.replace('global/', 'online/')
                variables[new_name] = var

            for v in self.variables_to_restore:
                self.sess.run(v.assign(variables[v.name.split(':')[0]]))

    def save_checkpoint(self):
        # only the chief process saves checkpoints
        if self.task_parameters.save_checkpoint_secs \
                and time.time() - self.last_checkpoint_saving_time >= self.task_parameters.save_checkpoint_secs \
                and (self.task_parameters.task_index == 0  # distributed
                     or self.task_parameters.task_index is None  # single-worker
                     ):

            checkpoint_path = os.path.join(self.task_parameters.save_checkpoint_dir,
                                           "{}_Step-{}.ckpt".format(
                                               self.checkpoint_id,
                                               self.total_steps_counters[RunPhase.TRAIN][EnvironmentSteps]))
            if not isinstance(self.task_parameters, DistributedTaskParameters):
                saved_checkpoint_path = self.checkpoint_saver.save(self.sess, checkpoint_path)
            else:
                saved_checkpoint_path = checkpoint_path

            # this is required in order for agents to save additional information like a DND for example
            [manager.save_checkpoint(self.checkpoint_id) for manager in self.level_managers]

            screen.log_dict(
                OrderedDict([
                    ("Saving in path", saved_checkpoint_path),
                ]),
                prefix="Checkpoint"
            )

            self.checkpoint_id += 1
            self.last_checkpoint_saving_time = time.time()

    def improve(self):
        """
        The main loop of the run.
        Defined in the following steps:
        1. Heatup
        2. Repeat:
            2.1. Repeat:
                2.1.1. Act
                2.1.2. Train
                2.1.3. Possibly save checkpoint
            2.2. Evaluate
        :return: None
        """

        # initialize the network parameters from the global network
        self.sync_graph()

        # heatup
        self.heatup(self.heatup_steps)

        # improve
        if self.task_parameters.task_index is not None:
            screen.log_title("Starting to improve {} task index {}".format(self.name, self.task_parameters.task_index))
        else:
            screen.log_title("Starting to improve {}".format(self.name))
        self.training_start_time = time.time()
        count_end = self.improve_steps.num_steps
        while self.total_steps_counters[RunPhase.TRAIN][self.improve_steps.__class__] < count_end:
            self.train_and_act(self.steps_between_evaluation_periods)
            self.evaluate(self.evaluation_steps)

    def __str__(self):
        result = ""
        for key, val in self.__dict__.items():
            params = ""
            if isinstance(val, list) or isinstance(val, dict) or isinstance(val, OrderedDict):
                items = iterable_to_items(val)
                for k, v in items:
                    params += "{}: {}\n".format(k, v)
            else:
                params = val
            result += "{}: \n{}\n".format(key, params)

        return result
