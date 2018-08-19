from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.architectures.tensorflow_components.middlewares.lstm_middleware import LSTMMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.environment import SingleLevelSelection, SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, AtariInputFilter
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(10000)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.05

agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].middleware_parameters = LSTMMiddlewareParameters(scheme=MiddlewareScheme.Medium,
                                                                                       number_of_lstm_cells=256)
agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'pong', 'space_invaders']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
