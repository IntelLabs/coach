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
this rollout worker:

- restores a model from disk
- evaluates a predefined number of episodes
- contributes them to a distributed memory
- exits
"""

from rl_coach.base_parameters import DistributedCoachSynchronizationType
from rl_coach.core_types import RunPhase


def rollout_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    if (
        graph_manager.agent_params.algorithm.distributed_coach_synchronization_type
        == DistributedCoachSynchronizationType.SYNC
    ):
        timeout = float("inf")
    else:
        timeout = None

    # this could probably be moved up into coach.py
    graph_manager.create_graph(task_parameters)

    data_store.load_policy(graph_manager, require_new_policy=False, timeout=60)

    with graph_manager.phase_context(RunPhase.TRAIN):
        # this worker should play a fraction of the total playing steps per rollout
        act_steps = (
            graph_manager.agent_params.algorithm.num_consecutive_playing_steps
            / num_workers
        )
        for i in range(graph_manager.improve_steps / act_steps):
            if data_store.end_of_policies():
                break

            graph_manager.act(
                act_steps,
                wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes,
            )

            data_store.load_policy(graph_manager, require_new_policy=True, timeout=timeout)
