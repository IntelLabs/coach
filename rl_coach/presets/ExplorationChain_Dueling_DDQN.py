from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

N = 20
num_output_head_copies = 20

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(2000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(N)

####################
# DQN Agent Params #
####################
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
agent_params.memory.max_size = (MemoryGranularity.Transitions, 1000000)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.1, (N+7)*2000)
agent_params.input_filter = NoInputFilter()
agent_params.output_filter = NoOutputFilter()

###############
# Environment #
###############
env_params = GymEnvironmentParameters(level='rl_coach.environments.toy_problems.exploration_chain:ExplorationChain')
env_params.additional_simulator_parameters = {'chain_length': N, 'max_steps': N+7}

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
