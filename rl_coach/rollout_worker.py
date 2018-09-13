import argparse

from rl_coach.base_parameters import TaskParameters
from rl_coach.coach import expand_preset
from rl_coach.core_types import EnvironmentEpisodes, RunPhase
from rl_coach.utils import short_dynamic_import



# TODO: acce[t preset option
# TODO: workers might need to define schedules in terms which can be synchronized: exploration(len(distributed_memory)) -> float
# TODO: periodically reload policy (from disk?)
# TODO: specify alternative distributed memory, or should this go in the preset?

def rollout_worker(graph_manager):
    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_dir='/checkpoint'
    graph_manager.create_graph(task_parameters)
    graph_manager.phase = RunPhase.TRAIN
    graph_manager.act(EnvironmentEpisodes(num_steps=10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str)
    args = parser.parse_args()

    graph_manager = short_dynamic_import(expand_preset(args.preset), ignore_module_case=True)
    rollout_worker(graph_manager)

if __name__ == '__main__':
    main()
