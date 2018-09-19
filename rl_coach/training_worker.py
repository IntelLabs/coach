"""
"""
import argparse
import time

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach import core_types
from rl_coach.utils import short_dynamic_import
from rl_coach.memories.non_episodic.distributed_experience_replay import DistributedExperienceReplay
from rl_coach.memories.memory import MemoryGranularity

# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take a non-distributed preset and automatically distribute it.

def heatup(graph_manager):
    memory = DistributedExperienceReplay(max_size=(MemoryGranularity.Transitions, 1000000),
                                         redis_ip=graph_manager.agent_params.memory.redis_ip,
                                         redis_port=graph_manager.agent_params.memory.redis_port)

    num_steps = graph_manager.schedule_params.heatup_steps.num_steps
    while(memory.num_transitions() < num_steps):
        time.sleep(10)


def training_worker(graph_manager, checkpoint_dir):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    # initialize graph
    task_parameters = TaskParameters()
    task_parameters.__dict__['save_checkpoint_dir'] = checkpoint_dir
    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # optionally wait for a specific number of transitions to be in memory before training
    heatup(graph_manager)

    # training loop
    for _ in range(40):
        graph_manager.phase = core_types.RunPhase.TRAIN
        graph_manager.train(core_types.TrainingSteps(1))
        graph_manager.phase = core_types.RunPhase.UNDEFINED

        graph_manager.evaluate(graph_manager.evaluation_steps)

        graph_manager.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint_dir',
                        help='(string) Path to a folder containing a checkpoint to write the model to.',
                        type=str,
                        default='/checkpoint')
    parser.add_argument('-r', '--redis_ip',
                        help="(string) IP or host for the redis server",
                        default='localhost',
                        type=str)
    parser.add_argument('-rp', '--redis_port',
                        help="(int) Port of the redis server",
                        default=6379,
                        type=int)
    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    graph_manager.agent_params.memory.redis_ip = args.redis_ip
    graph_manager.agent_params.memory.redis_port = args.redis_port

    training_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
