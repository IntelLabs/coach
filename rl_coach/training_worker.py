"""
"""
import argparse

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach import core_types
from rl_coach.utils import short_dynamic_import

# Q: specify alternative distributed memory, or should this go in the preset?
# A: preset must define distributed memory to be used. we aren't going to take a non-distributed preset and automatically distribute it.


def heatup(graph_manager):
    num_steps = graph_manager.schedule_params.heatup_steps.num_steps
    while len(graph_manager.agent_params.memory) < num_steps:
        time.sleep(1)


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

    # TODO: Q: training steps passed into graph_manager.train ignored?
    # TODO: specify training steps between checkpoints (in preset?)
    # TODO: replace outer training loop with something general
    # TODO: low priority: move evaluate out of this process

    heatup(graph_manager)

    # training loop
    for _ in range(10):
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
    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)

    training_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
    )

if __name__ == '__main__':
    main()
