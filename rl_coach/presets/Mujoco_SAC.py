from rl_coach.agents.soft_actor_critic_agent import SoftActorCriticAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

########################
# my_sac configuration #
########################

# some info about the original SAC implementation:
# max_replay_buffer_size = 1e6
# sampler parameters:
#   batch_size = 256            # in each training iteration we have batch_size transitions in the batch
#   max_path_length = 1000      # the maximum number of timesteps allowed per path after which the environment is reset
#   min_pool_size = 1000        # minimum number of timesteps that should be in the buffer before we can sample a batch
# algorithm parameters:
#   discount = 0.99             # the reward discount factor
#   lr = 3e-4                   # the learning rate that is used to update all networks
#   scale_reward = 5 or 20      # the factor to multiply the reward
#   tau = 0.005                 # soft-update rate of value target network: (1-tau)*target+tau*online
#   target_update_interval = 1  # number of iterations between soft updates of target network (1=every iteration)
#   epoch_length = 1000         # epoch_length * n_epochs = number of training iterations (steps)
#   n_epochs = 3000             # during an epoch we add one sample to the replay buffer and then do (sample a batch
                                # for training --> learn from batch) iteration for n_train_repeat times
                                # at the end of each epoch we perform policy evaluation
#   n_train_repeats = 1         # number of training iterations (batches) to train per each sample added to buffer
#   eval_deterministic = True   # whether to use deterministic or stochastic policy for evaluation
#   eval_n_episodes = 1         # number of episodes to rollout during evaluation
#   n_initial_exploration_steps = 10000     # number of steps to have with initial exploration policy (if defined)
#   eval_render = False         # whether to render during evaluation

#------------------
# mapping to coach parameters
# batch_size  --> NetworkParameters.batch_size
# max_path_length --> ???
# min_pool_size --> schedule_params.heatup_steps
# discount --> AlgorithmParameters.discount
# lr --> NetworkParameters.learning_rate
# scale_reward --> agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(5))
# target_update_interval --> AlgorithmParameters.num_steps_between_copying_online_weights_to_target
# tau --> AlgorithmParameters.rate_for_copying_weights_to_target
# epoch_length --> schedule_params.steps_between_evaluation_periods = EnvironmentSteps(epoch_length)
# n_epochs --> schedule_params.improve_steps ???
# n_train_repeats --> AlgorithmParameters.num_consecutive_training_steps
# eval_deterministic --> SAC AlgorithmParameters.use_deterministic_for_evaluation
# eval_n_episodes --> schedule_params.evaluation_steps = EnvironmentEpisodes(1) ? check it !
# n_initial_exploration_steps --> schedule_params.heatup_steps ( with algorithm.heatup_using_network_decisions = False)
# eval_render -->





####################
# Graph Scheduling #

####################

# from DDPG (that is using EpisodicExperienceReplay)
# schedule_params = ScheduleParameters()
# schedule_params.improve_steps = EnvironmentSteps(2000000)
# schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
# schedule_params.evaluation_steps = EnvironmentEpisodes(1)
# schedule_params.heatup_steps = EnvironmentSteps(1000)





# see graph_manager.py for possible schedule parameters
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(3000000)
# schedule_params.improve_steps = EnvironmentSteps(1000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(1000)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(10000)




#########
# Agent #
#########
agent_params = SoftActorCriticAgentParameters()
# override default parameters:
# value (v) networks parameters
agent_params.network_wrappers['v'].batch_size = 256
agent_params.network_wrappers['v'].learning_rate = 0.0003
agent_params.network_wrappers['v'].middleware_parameters.scheme = [Dense(256)]

# critic (q) network parameters
agent_params.network_wrappers['q'].heads_parameters[0].network_layers_sizes = (256,256)
agent_params.network_wrappers['q'].batch_size = 256
agent_params.network_wrappers['q'].learning_rate = 0.0003

# actor (policy) network parameters
agent_params.network_wrappers['policy'].batch_size = 256
agent_params.network_wrappers['policy'].learning_rate = 0.0003
agent_params.network_wrappers['policy'].middleware_parameters.scheme = [Dense(256)]

# Algorithm params



# Input Filter
# SAC requires reward scaling for Mujoco environments.
# according to the paper:
# Hopper, Walker-2d, HalfCheetah, Ant - requires scaling of 5
# Humanoid - requires scaling of 20

# disabled for debug only
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter('rescale', RewardRescaleFilter(5))

###############
# Environment #
###############
env_params = GymVectorEnvironment(level=SingleLevelSelection(mujoco_v2))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 400
preset_validation_params.max_episodes_to_achieve_reward = 1000
preset_validation_params.reward_test_level = 'inverted_pendulum'
preset_validation_params.trace_test_levels = ['inverted_pendulum', 'hopper']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
