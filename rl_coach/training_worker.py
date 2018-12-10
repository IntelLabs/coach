"""
"""
import time

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType
from rl_coach import core_types
from rl_coach.logger import screen


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
            # screen.log_title('steps: ' + str(steps))
            # screen.log_title('improve steps: ' + str(graph_manager.improve_steps.num_steps/math.ceil((graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps)/num_workers)))

            graph_manager.phase = core_types.RunPhase.TRAIN
            graph_manager.train()
            graph_manager.phase = core_types.RunPhase.UNDEFINED
            screen.log_title('achieved: ' + str(graph_manager.achieved_success_rate))
            if graph_manager.achieved_success_rate:
                break
            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                graph_manager.save_checkpoint()
            else:
                graph_manager.occasionally_save_checkpoint()
