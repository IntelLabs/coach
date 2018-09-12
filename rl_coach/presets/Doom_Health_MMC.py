from rl_coach.agents.mmc_agent import MixedMonteCarloAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.doom_environment import DoomEnvironmentParameters
from rl_coach.environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(5)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = MixedMonteCarloAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.exploration.epsilon_schedule = LinearSchedule(0.5, 0, 10000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.memory.max_size = (MemoryGranularity.Episodes, 200)
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False


###############
# Environment #
###############
env_params = DoomEnvironmentParameters()
env_params.level = 'HEALTH_GATHERING'
vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test_using_a_trace_test = False
# disabling this test for now, as it takes too long to converge
# preset_validation_params.test = True
# preset_validation_params.min_reward_threshold = 1000
# preset_validation_params.max_episodes_to_achieve_reward = 300

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
