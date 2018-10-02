from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme, PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, atari_schedule
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from rl_coach.schedules import LinearSchedule, PieceWiseSchedule, ConstantSchedule


#########
# Agent #
#########
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.network_wrappers['main'].clip_gradients = 10
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(40000)
agent_params.exploration.epsilon_schedule = PieceWiseSchedule(
    [(LinearSchedule(1, 0.1, 1000000), EnvironmentSteps(1000000)),
     (LinearSchedule(0.1, 0.01, 10000000), EnvironmentSteps(1000000)),
     (ConstantSchedule(0.001), EnvironmentSteps(10000000))]
)
agent_params.memory = PrioritizedExperienceReplayParameters()
agent_params.memory.beta = LinearSchedule(0.4, 1, 12500000)  # 12.5M training iterations = 50M steps = 200M frames

###############
# Environment #
###############
env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'pong', 'space_invaders']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=atari_schedule, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
