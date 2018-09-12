from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentEpisodes, RunPhase
from rl_coach.presets.CartPole_DQN import graph_manager


# TODO: workers might need to define schedules in terms which can be synchronized: exploration(len(distributed_memory)) -> float

def main():
    graph_manager.create_graph(TaskParameters())
    graph_manager.phase = RunPhase.TRAIN
    graph_manager.act(EnvironmentEpisodes(num_steps=10))

if __name__ == '__main__':
    main()
