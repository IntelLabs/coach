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


"""
"""
import time

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType
from rl_coach import core_types
from rl_coach.logger import screen


def data_store_ckpt_load(data_store):
    if data_store:
        data_store.load_from_store()


def training_worker(graph_manager, task_parameters, data_store, is_multi_node_test):
    """
    restore a checkpoint then perform rollouts using the restored model
    :param graph_manager: An instance of the graph manager
    :param task_parameters: An instance of task parameters
    :param is_multi_node_test: If this is a multi node test insted of a normal run.
    """
    # Load checkpoint if provided
    if task_parameters.checkpoint_restore_path:
        data_store_ckpt_load(data_store)
        # initialize graph
        graph_manager.create_graph(task_parameters)

    else:
        # initialize graph
        graph_manager.create_graph(task_parameters)

        # save randomly initialized graph
        graph_manager.save_checkpoint()

    # training loop
    steps = 0

    # evaluation offset
    eval_offset = 1

    graph_manager.setup_memory_backend()
    graph_manager.signal_ready()

    while steps < graph_manager.improve_steps.num_steps:

        graph_manager.phase = core_types.RunPhase.TRAIN
        if is_multi_node_test and graph_manager.get_current_episodes_count() > graph_manager.preset_validation_params.max_episodes_to_achieve_reward:
            # Test failed as it has not reached the required success rate
            graph_manager.flush_finished()
            screen.error("Could not reach required success by {} episodes.".format(graph_manager.preset_validation_params.max_episodes_to_achieve_reward), crash=True)

        graph_manager.fetch_from_worker(graph_manager.agent_params.algorithm.num_consecutive_playing_steps)
        graph_manager.phase = core_types.RunPhase.UNDEFINED

        if graph_manager.should_train():
            steps += 1

            graph_manager.phase = core_types.RunPhase.TRAIN
            graph_manager.train()
            graph_manager.phase = core_types.RunPhase.UNDEFINED

            if steps * graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps > graph_manager.steps_between_evaluation_periods.num_steps * eval_offset:
                eval_offset += 1
                if graph_manager.evaluate(graph_manager.evaluation_steps):
                    break

            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                graph_manager.save_checkpoint()
            else:
                graph_manager.occasionally_save_checkpoint()
