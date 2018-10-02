from rl_coach.agents.dfp_agent import DFPAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.doom_environment import DoomEnvironmentParameters, DoomEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(6250000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(62500)
schedule_params.evaluation_steps = EnvironmentSteps(6250)
schedule_params.heatup_steps = EnvironmentSteps(1)

#########
# Agent #
#########
agent_params = DFPAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0001
# the original DFP code decays  epsilon in ~1.5M steps. Only that unlike other most other papers, these are 1.5M
# training steps. i.e. it is equivalent to once every 8 playing steps (when a training batch is sampled).
# so this is 1.5M*8 =~ 12M playing steps per worker.
# TODO allow the epsilon schedule to be defined in terms of training steps.
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0, 12000000)
agent_params.exploration.evaluation_epsilon = 0
agent_params.algorithm.use_accumulated_reward_as_measurement = False
agent_params.algorithm.goal_vector = [0.5, 0.5, 1]  # ammo, health, frag count
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].input_rescaling['vector'] = 100.
agent_params.algorithm.scale_measurements_targets['GameVariable.HEALTH'] = 30.0
agent_params.algorithm.scale_measurements_targets['GameVariable.AMMO2'] = 7.5
agent_params.algorithm.scale_measurements_targets['GameVariable.USER2'] = 1.0
agent_params.network_wrappers['main'].learning_rate_decay_rate = 0.3
agent_params.network_wrappers['main'].learning_rate_decay_steps = 250000
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'].input_offset['vector'] = 0.5
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].input_offset['vector'] = 0.5


###############
# Environment #
###############
env_params = DoomEnvironmentParameters(level='BATTLE_COACH_LOCAL')
env_params.cameras = [DoomEnvironment.CameraTypes.OBSERVATION]


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
