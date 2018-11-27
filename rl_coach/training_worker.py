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


def data_store_ckpt_save(data_store):
    while True:
        data_store.save_to_store()
        time.sleep(10)


def training_worker(graph_manager, task_parameters):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    # initialize graph
    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # training loop
    steps = 0

    # evaluation offset
    eval_offset = 1

    graph_manager.setup_memory_backend()

    while steps < graph_manager.improve_steps.num_steps:

        graph_manager.phase = core_types.RunPhase.TRAIN
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
