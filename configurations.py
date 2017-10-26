#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from utils import Enum
import json
from logger import screen, logger


class Frameworks(Enum):
    TensorFlow = 1
    Neon = 2


class InputTypes(object):
    Observation = 1
    Measurements = 2
    GoalVector = 3
    Action = 4
    TimedObservation = 5


class OutputTypes(object):
    Q = 1
    DuelingQ = 2
    V = 3
    Pi = 4
    MeasurementsPrediction = 5
    DNDQ = 6
    NAF = 7
    PPO = 8
    PPO_V = 9
    DistributionalQ = 10


class MiddlewareTypes(object):
    LSTM = 1
    FC = 2


class AgentParameters(object):
    agent = ''

    # Architecture parameters
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Q]
    middleware_type = MiddlewareTypes.FC
    loss_weights = [1.0]
    stop_gradients_from_head = [False]
    num_output_head_copies = 1
    use_measurements = False
    use_accumulated_reward_as_measurement = False
    add_a_normalized_timestep_to_the_observation = False
    l2_regularization = 0
    hidden_layers_activation_function = 'relu'
    optimizer_type = 'Adam'
    async_training = False
    use_separate_networks_per_head = False

    # Agent parameters
    num_consecutive_playing_steps = 1
    num_consecutive_training_steps = 1
    bootstrap_total_return_from_old_policy = False
    n_step = -1
    num_episodes_in_experience_replay = 200
    num_transitions_in_experience_replay = None
    discount = 0.99
    policy_gradient_rescaler = 'A_VALUE'
    apply_gradients_every_x_episodes = 5
    beta_entropy = 0
    num_steps_between_gradient_updates = 20000  # t_max
    num_steps_between_copying_online_weights_to_target = 1000
    rate_for_copying_weights_to_target = 1.0
    monte_carlo_mixing_rate = 0.1
    gae_lambda = 0.96
    step_until_collecting_full_episodes = False
    targets_horizon = 'N-Step'
    replace_mse_with_huber_loss = False

    # PPO related params
    target_kl_divergence = 0.01
    initial_kl_coefficient = 1.0
    high_kl_penalty_coefficient = 1000
    value_targets_mix_fraction = 0.1
    clip_likelihood_ratio_using_epsilon = None
    use_kl_regularization = True
    estimate_value_using_gae = False

    # DFP related params
    num_predicted_steps_ahead = 6
    goal_vector = [1.0, 1.0]
    future_measurements_weights = [0.5, 0.5, 1.0]

    # NEC related params
    dnd_size = 500000
    l2_norm_added_delta = 0.001
    new_value_shift_coefficient = 0.1
    number_of_knn = 50
    DND_key_error_threshold = 0.01

    # Framework support
    neon_support = False
    tensorflow_support = True

    # distributed agents params
    shared_optimizer = True
    share_statistics_between_workers = True


class EnvironmentParameters(object):
    type = 'Doom'
    level = 'basic'
    observation_stack_size = 4
    frame_skip = 4
    desired_observation_width = 76
    desired_observation_height = 60
    normalize_observation = False
    reward_scaling = 1.0
    reward_clipping_min = None
    reward_clipping_max = None


class ExplorationParameters(object):
    # Exploration policies
    policy = 'EGreedy'
    evaluation_policy = 'Greedy'
    # -- bootstrap dqn parameters
    bootstrapped_data_sharing_probability = 0.5
    architecture_num_q_heads = 1
    # -- dropout approximation of thompson sampling parameters
    dropout_discard_probability = 0
    initial_keep_probability = 0.0  # unused
    final_keep_probability = 0.99  # unused
    keep_probability_decay_steps = 50000  # unused
    # -- epsilon greedy parameters
    initial_epsilon = 0.5
    final_epsilon = 0.01
    epsilon_decay_steps = 50000
    evaluation_epsilon = 0.05
    # -- epsilon greedy at end of episode parameters
    average_episode_length_over_num_episodes = 20
    # -- boltzmann softmax parameters
    initial_temperature = 100.0
    final_temperature = 1.0
    temperature_decay_steps = 50000
    # -- additive noise
    initial_noise_variance_percentage = 0.1
    final_noise_variance_percentage = 0.1
    noise_variance_decay_steps = 1
    # -- Ornstein-Uhlenbeck process
    mu = 0
    theta = 0.15
    sigma = 0.3
    dt = 0.01


class GeneralParameters(object):
    train = True
    framework = Frameworks.TensorFlow
    threads = 1
    sess = None

    # distributed training options
    num_threads = 1
    synchronize_over_num_threads = 1
    distributed = False

    # Agent blocks
    memory = 'EpisodicExperienceReplay'
    architecture = 'GeneralTensorFlowNetwork'

    # General parameters
    clip_gradients = None
    kl_divergence_constraint = 100000
    num_training_iterations = 10000000000
    num_heatup_steps = 1000
    batch_size = 32
    save_model_sec = None
    save_model_dir = None
    checkpoint_restore_dir = None
    learning_rate = 0.00025
    learning_rate_decay_rate = 0
    learning_rate_decay_steps = 0
    evaluation_episodes = 5
    evaluate_every_x_episodes = 1000000
    rescaling_interpolation_type = 'bilinear'

    # setting a seed will only work for non-parallel algorithms. Parallel algorithms add uncontrollable noise in
    # the form of different workers starting at different times, and getting different assignments of CPU
    # time from the OS.
    seed = None

    checkpoints_path = ''

    # Testing parameters
    test = False
    test_min_return_threshold = 0
    test_max_step_threshold = 1
    test_num_workers = 1


class VisualizationParameters(object):
    # Visualization parameters
    record_video_every = 1000
    video_path = '/home/llt_lab/temp/breakout-videos'
    plot_action_values_online = False
    show_saliency_maps_every_num_episodes = 1000000000
    print_summary = False
    dump_csv = True
    dump_signals_to_csv_every_x_episodes = 10
    render = False
    dump_gifs = True


class Roboschool(EnvironmentParameters):
    type = 'Gym'
    frame_skip = 1
    observation_stack_size = 1
    desired_observation_height = None
    desired_observation_width = None


class GymVectorObservation(EnvironmentParameters):
    type = 'Gym'
    frame_skip = 1
    observation_stack_size = 1
    desired_observation_height = None
    desired_observation_width = None


class Bullet(EnvironmentParameters):
    type = 'Bullet'
    frame_skip = 1
    observation_stack_size = 1
    desired_observation_height = None
    desired_observation_width = None


class Atari(EnvironmentParameters):
    type = 'Gym'
    frame_skip = 1
    observation_stack_size = 4
    desired_observation_height = 84
    desired_observation_width = 84
    reward_clipping_max = 1.0
    reward_clipping_min = -1.0


class Doom(EnvironmentParameters):
    type = 'Doom'
    frame_skip = 4
    observation_stack_size = 3
    desired_observation_height = 60
    desired_observation_width = 76


class NStepQ(AgentParameters):
    type = 'NStepQAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Q]
    loss_weights = [1.0]
    optimizer_type = 'Adam'
    num_steps_between_copying_online_weights_to_target = 1000
    num_episodes_in_experience_replay = 2
    apply_gradients_every_x_episodes = 1
    num_steps_between_gradient_updates = 20  # this is called t_max in all the papers
    hidden_layers_activation_function = 'elu'
    targets_horizon = 'N-Step'
    async_training = True
    shared_optimizer = True


class DQN(AgentParameters):
    type = 'DQNAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Q]
    loss_weights = [1.0]
    optimizer_type = 'Adam'
    num_steps_between_copying_online_weights_to_target = 1000
    neon_support = True
    async_training = True
    shared_optimizer = True


class DDQN(DQN):
    type = 'DDQNAgent'

class DuelingDQN(DQN):
    type = 'DQNAgent'
    output_types = [OutputTypes.DuelingQ]

class BootstrappedDQN(DQN):
    type = 'BootstrappedDQNAgent'
    num_output_head_copies = 10


class DistributionalDQN(DQN):
    type = 'DistributionalDQNAgent'
    output_types = [OutputTypes.DistributionalQ]
    v_min = -10.0
    v_max = 10.0
    atoms = 51


class NEC(AgentParameters):
    type = 'NECAgent'
    optimizer_type = 'RMSProp'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.DNDQ]
    loss_weights = [1.0]
    dnd_size = 500000
    l2_norm_added_delta = 0.001
    new_value_shift_coefficient = 0.1
    number_of_knn = 50
    n_step = 100
    bootstrap_total_return_from_old_policy = True
    DND_key_error_threshold = 0.1


class ActorCritic(AgentParameters):
    type = 'ActorCriticAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.V, OutputTypes.Pi]
    loss_weights = [0.5, 1.0]
    stop_gradients_from_head = [False, False]
    num_episodes_in_experience_replay = 2
    policy_gradient_rescaler = 'A_VALUE'
    hidden_layers_activation_function = 'elu'
    apply_gradients_every_x_episodes = 5
    beta_entropy = 0
    num_steps_between_gradient_updates = 5000  # this is called t_max in all the papers
    gae_lambda = 0.96
    shared_optimizer = True
    estimate_value_using_gae = False
    async_training = True


class PolicyGradient(AgentParameters):
    type = 'PolicyGradientsAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Pi]
    loss_weights = [1.0]
    num_episodes_in_experience_replay = 2
    policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
    apply_gradients_every_x_episodes = 5
    beta_entropy = 0
    num_steps_between_gradient_updates = 20000  # this is called t_max in all the papers
    async_training = True


class DDPG(AgentParameters):
    type = 'DDPGAgent'
    input_types = [InputTypes.Observation, InputTypes.Action]
    output_types = [OutputTypes.V]  # V is used because we only want a single Q value
    loss_weights = [1.0]
    hidden_layers_activation_function = 'relu'
    num_episodes_in_experience_replay = 10000
    num_steps_between_copying_online_weights_to_target = 1
    rate_for_copying_weights_to_target = 0.001
    shared_optimizer = True
    async_training = True


class DDDPG(AgentParameters):
    type = 'DDPGAgent'
    input_types = [InputTypes.Observation, InputTypes.Action]
    output_types = [OutputTypes.V]  # V is used because we only want a single Q value
    loss_weights = [1.0]
    hidden_layers_activation_function = 'relu'
    num_episodes_in_experience_replay = 10000
    num_steps_between_copying_online_weights_to_target = 10
    rate_for_copying_weights_to_target = 1
    shared_optimizer = True
    async_training = True


class NAF(AgentParameters):
    type = 'NAFAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.NAF]
    loss_weights = [1.0]
    hidden_layers_activation_function = 'tanh'
    num_consecutive_training_steps = 5
    num_steps_between_copying_online_weights_to_target = 1
    rate_for_copying_weights_to_target = 0.001
    optimizer_type = 'RMSProp'
    async_training = True


class PPO(AgentParameters):
    type = 'PPOAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.V]
    loss_weights = [1.0]
    hidden_layers_activation_function = 'tanh'
    num_episodes_in_experience_replay = 1000000
    policy_gradient_rescaler = 'A_VALUE'
    gae_lambda = 0.96
    target_kl_divergence = 0.01
    initial_kl_coefficient = 1.0
    high_kl_penalty_coefficient = 1000
    add_a_normalized_timestep_to_the_observation = True
    l2_regularization = 0#1e-3
    value_targets_mix_fraction = 0.1
    async_training = True
    estimate_value_using_gae = True
    step_until_collecting_full_episodes = True


class ClippedPPO(AgentParameters):
    type = 'ClippedPPOAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.V, OutputTypes.PPO]
    loss_weights = [0.5, 1.0]
    stop_gradients_from_head = [False, False]
    hidden_layers_activation_function = 'tanh'
    num_episodes_in_experience_replay = 1000000
    policy_gradient_rescaler = 'GAE'
    gae_lambda = 0.95
    target_kl_divergence = 0.01
    initial_kl_coefficient = 1.0
    high_kl_penalty_coefficient = 1000
    add_a_normalized_timestep_to_the_observation = False
    l2_regularization = 1e-3
    value_targets_mix_fraction = 0.1
    clip_likelihood_ratio_using_epsilon = 0.2
    async_training = False
    use_kl_regularization = False
    estimate_value_using_gae = True
    batch_size = 64
    use_separate_networks_per_head = True
    step_until_collecting_full_episodes = True
    beta_entropy = 0.01

class DFP(AgentParameters):
    type = 'DFPAgent'
    input_types = [InputTypes.Observation, InputTypes.Measurements, InputTypes.GoalVector]
    output_types = [OutputTypes.MeasurementsPrediction]
    loss_weights = [1.0]
    use_measurements = True
    num_predicted_steps_ahead = 6
    goal_vector = [1.0, 1.0]
    future_measurements_weights = [0.5, 0.5, 1.0]
    async_training = True


class MMC(AgentParameters):
    type = 'MixedMonteCarloAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Q]
    loss_weights = [1.0]
    num_steps_between_copying_online_weights_to_target = 1000
    monte_carlo_mixing_rate = 0.1
    neon_support = True


class PAL(AgentParameters):
    type = 'PALAgent'
    input_types = [InputTypes.Observation]
    output_types = [OutputTypes.Q]
    loss_weights = [1.0]
    pal_alpha = 0.9
    persistent_advantage_learning = False
    num_steps_between_copying_online_weights_to_target = 1000
    neon_support = True


class EGreedyExploration(ExplorationParameters):
    policy = 'EGreedy'
    initial_epsilon = 0.5
    final_epsilon = 0.01
    epsilon_decay_steps = 50000
    evaluation_epsilon = 0.05
    initial_noise_variance_percentage = 0.1
    final_noise_variance_percentage = 0.1
    noise_variance_decay_steps = 50000


class BootstrappedDQNExploration(ExplorationParameters):
    policy = 'Bootstrapped'
    architecture_num_q_heads = 10
    bootstrapped_data_sharing_probability = 0.1


class OUExploration(ExplorationParameters):
    policy = 'OUProcess'
    mu = 0
    theta = 0.15
    sigma = 0.3
    dt = 0.01


class AdditiveNoiseExploration(ExplorationParameters):
    policy = 'AdditiveNoise'
    initial_noise_variance_percentage = 0.1
    final_noise_variance_percentage = 0.1
    noise_variance_decay_steps = 50000


class EntropyExploration(ExplorationParameters):
    policy = 'ContinuousEntropy'


class CategoricalExploration(ExplorationParameters):
    policy = 'Categorical'


class Preset(GeneralParameters):
    def __init__(self, agent, env, exploration, visualization=VisualizationParameters):
        """
        :type agent: AgentParameters
        :type env: EnvironmentParameters
        :type exploration: ExplorationParameters
        :type visualization: VisualizationParameters
        """
        self.visualization = visualization
        self.agent = agent
        self.env = env
        self.exploration = exploration

