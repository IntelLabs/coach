from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, RunPhase
from rl_coach.environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(50000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(1000000)
schedule_params.evaluation_steps = EnvironmentSteps(125000)
schedule_params.heatup_steps = EnvironmentSteps(20000)

#########
# Agent #
#########
agent_params = RainbowDQNAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0000625
agent_params.network_wrappers['main'].optimizer_epsilon = 1.5e-4
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(32000 // 4)  # 32k frames
agent_params.memory.beta = LinearSchedule(0.4, 1, 12500000)  # 12.5M training iterations = 50M steps = 200M frames
agent_params.memory.alpha = 0.5

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'pong', 'space_invaders']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
