import copy

from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters, CameraTypes, CarlaInputFilter
from rl_coach.environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DDPGAgentParameters()
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)

# front camera
agent_params.network_wrappers['actor'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['actor'].input_embedders_parameters.pop('observation')
agent_params.network_wrappers['critic'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['critic'].input_embedders_parameters.pop('observation')

# left camera
agent_params.network_wrappers['actor'].input_embedders_parameters['left_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['actor'].input_embedders_parameters['forward_camera'])
agent_params.network_wrappers['critic'].input_embedders_parameters['left_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['critic'].input_embedders_parameters['forward_camera'])

# right camera
agent_params.network_wrappers['actor'].input_embedders_parameters['right_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['actor'].input_embedders_parameters['forward_camera'])
agent_params.network_wrappers['critic'].input_embedders_parameters['right_camera'] = \
    copy.deepcopy(agent_params.network_wrappers['critic'].input_embedders_parameters['forward_camera'])

agent_params.input_filter = CarlaInputFilter()
agent_params.input_filter.copy_filters_from_one_observation_to_another('forward_camera', 'left_camera')
agent_params.input_filter.copy_filters_from_one_observation_to_another('forward_camera', 'right_camera')

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.level = 'town1'
env_params.cameras = [CameraTypes.FRONT, CameraTypes.LEFT, CameraTypes.RIGHT]

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
