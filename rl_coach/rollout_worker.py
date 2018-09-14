"""
this rollout worker restores a model from disk, evaluates a predefined number of
episodes, and contributes them to a distributed memory
"""
import argparse

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach.core_types import EnvironmentEpisodes, RunPhase
from rl_coach.utils import short_dynamic_import

# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take a non-distributed preset and automatically distribute it.


def rollout_worker(graph_manager, checkpoint_dir):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_restore_dir'] = checkpoint_dir
    graph_manager.create_graph(task_parameters)
    graph_manager.phase = RunPhase.TRAIN
    graph_manager.act(EnvironmentEpisodes(num_steps=10))
    graph_manager.phase = RunPhase.UNDEFINED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('--checkpoint_dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str,
                        default='/checkpoint')
    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    rollout_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )

if __name__ == '__main__':
    main()
