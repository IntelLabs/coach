import math

from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme, PresetValidationParameters
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, atari_schedule
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

#########
# Agent #
#########
agent_params = DDQNAgentParameters()

# since we are using Adam instead of RMSProp, we adjust the learning rate as well
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty
agent_params.network_wrappers['main'].heads_parameters = \
    [DuelingQHeadParameters(rescale_gradient_from_head_by_factor=1/math.sqrt(2))]
agent_params.network_wrappers['main'].clip_gradients = 10

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
