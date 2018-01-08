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

from configurations import *
import ast
import sys


def json_to_preset(json_path):
    with open(json_path, 'r') as json_file:
        run_dict = json.loads(json_file.read())

    if run_dict['preset'] is None:
        tuning_parameters = Preset(eval(run_dict['agent_type']), eval(run_dict['environment_type']),
                                   eval(run_dict['exploration_policy_type']))
    else:
        tuning_parameters = eval(run_dict['preset'])()
        # Override existing parts of the preset
        if run_dict['agent_type'] is not None:
            tuning_parameters.agent = eval(run_dict['agent_type'])()

        if run_dict['environment_type'] is not None:
            tuning_parameters.env = eval(run_dict['environment_type'])()

        if run_dict['exploration_policy_type'] is not None:
            tuning_parameters.exploration = eval(run_dict['exploration_policy_type'])()

    # human control
    if run_dict['play']:
        tuning_parameters.agent.type = 'HumanAgent'
        tuning_parameters.env.human_control = True
        tuning_parameters.num_heatup_steps = 0
        
    if run_dict['level']:
        tuning_parameters.env.level = run_dict['level']

    if run_dict['custom_parameter'] is not None:
        unstripped_key_value_pairs = [pair.split('=') for pair in run_dict['custom_parameter'].split(';')]
        stripped_key_value_pairs = [tuple([pair[0].strip(), ast.literal_eval(pair[1].strip())]) for pair in
                                    unstripped_key_value_pairs]

        # load custom parameters into run_dict
        for key, value in stripped_key_value_pairs:
            run_dict[key] = value

    for key in ['agent_type', 'environment_type', 'exploration_policy_type', 'preset', 'custom_parameter']:
        run_dict.pop(key, None)

    # load parameters from run_dict to tuning_parameters
    for key, value in run_dict.items():
        if ((sys.version_info[0] == 2 and type(value) == unicode) or
                (sys.version_info[0] == 3 and type(value) == str)):
            value = '"{}"'.format(value)
        exec('tuning_parameters.{} = {}'.format(key, value)) in globals(), locals()

    return tuning_parameters


class Doom_Basic_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, DQN, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000


class Doom_Basic_QRDQN(Preset):
    def __init__(self):
        Preset.__init__(self, QuantileRegressionDQN, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000



class Doom_Basic_OneStepQ(Preset):
    def __init__(self):
        Preset.__init__(self, NStepQ, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 0
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.agent.optimizer_type = 'Adam'
        self.clip_gradients = 1000
        self.agent.targets_horizon = '1-Step'


class Doom_Basic_NStepQ(Preset):
    def __init__(self):
        Preset.__init__(self, NStepQ, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.learning_rate = 0.000025
        self.num_heatup_steps = 0
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.agent.optimizer_type = 'Adam'
        self.clip_gradients = 1000


class Doom_Basic_A2C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, Doom, CategoricalExploration)
        self.env.level = 'basic'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100
        self.env.reward_scaling = 100.


class Doom_Basic_Dueling_DDQN(Preset):
    def __init__(self):
        Preset.__init__(self, DDQN, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.output_types = [OutputTypes.DuelingQ]
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000

class Doom_Basic_Dueling_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, DuelingDQN, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000


class CartPole_Dueling_DDQN(Preset):
    def __init__(self):
        Preset.__init__(self, DDQN, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.output_types = [OutputTypes.DuelingQ]
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 100
        self.test_min_return_threshold = 150


class Doom_Health_MMC(Preset):
    def __init__(self):
        Preset.__init__(self, MMC, Doom, ExplorationParameters)
        self.env.level = 'HEALTH_GATHERING'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 10000

class CartPole_MMC(Preset):
    def __init__(self):
        Preset.__init__(self, MMC, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 90
        self.test_min_return_threshold = 150


class CartPole_PAL(Preset):
    def __init__(self):
        Preset.__init__(self, PAL, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 100
        self.test_min_return_threshold = 150

class Doom_Basic_DFP(Preset):
    def __init__(self):
        Preset.__init__(self, DFP, Doom, ExplorationParameters)
        self.env.level = 'BASIC'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.0001
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 10000
        self.agent.use_accumulated_reward_as_measurement = True
        self.agent.goal_vector = [0.0, 1.0]
        # self.agent.num_consecutive_playing_steps = 10


class Doom_Health_DFP(Preset):
    def __init__(self):
        Preset.__init__(self, DFP, Doom, ExplorationParameters)
        self.env.level = 'HEALTH_GATHERING'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 10000
        self.agent.use_accumulated_reward_as_measurement = True


class Doom_Deadly_Corridor_Bootstrapped_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, BootstrappedDQN, Doom, BootstrappedDQNExploration)
        self.env.level = 'deadly_corridor'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.agent.num_steps_between_copying_online_weights_to_target = 1000
        self.num_heatup_steps = 1000


class CartPole_Bootstrapped_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, BootstrappedDQN, GymVectorObservation, BootstrappedDQNExploration)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 200
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150

class CartPole_PG(Preset):
    def __init__(self):
        Preset.__init__(self, PolicyGradient, GymVectorObservation, CategoricalExploration)
        self.env.level = 'CartPole-v0'
        self.agent.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.001
        self.num_heatup_steps = 100
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 150
        self.test_min_return_threshold = 150


class CartPole_PPO(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, GymVectorObservation, CategoricalExploration)
        self.env.level = 'CartPole-v0'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 512
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'LBFGS'
        self.env.normalize_observation = True

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150

class CartPole_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, CategoricalExploration)
        self.env.level = 'CartPole-v0'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 512
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150

class CartPole_A2C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, CategoricalExploration)
        self.env.level = 'CartPole-v0'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 200.
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 300
        self.test_min_return_threshold = 150


class CartPole_OneStepQ(Preset):
    def __init__(self):
        Preset.__init__(self, NStepQ, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.discount = 1.0
        self.agent.targets_horizon = '1-Step'


class CartPole_NStepQ(Preset):
    def __init__(self):
        Preset.__init__(self, NStepQ, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.0001
        self.exploration.epsilon_decay_steps = 10000
        self.num_heatup_steps = 0
        self.agent.discount = 0.99
        self.agent.num_steps_between_gradient_updates = 5

        self.test = True
        self.test_max_step_threshold = 2000
        self.test_min_return_threshold = 150
        self.test_num_workers = 8

class CartPole_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, DQN, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 150
        self.test_min_return_threshold = 150


class CartPole_C51(Preset):
    def __init__(self):
        Preset.__init__(self, CategoricalDQN, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0
        # self.env.reward_scaling = 20.
        self.agent.v_min = 0.0
        self.agent.v_max = 200.0

        self.test = True
        self.test_max_step_threshold = 150
        self.test_min_return_threshold = 150


class CartPole_QRDQN(Preset):
    def __init__(self):
        Preset.__init__(self, QuantileRegressionDQN, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.agent.num_steps_between_copying_online_weights_to_target = 100
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 3000
        self.agent.discount = 1.0


# The below preset matches the hyper-parameters setting as in the original DQN paper.
# This a very resource intensive preset, and might easily blow up your RAM (> 100GB of usage).
# Try reducing the number of transitions in the experience replay (50e3 might be a reasonable number to start with),
# so to make sure it fits your RAM.
class Breakout_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, DQN, Atari, ExplorationParameters)
        self.env.level = 'BreakoutDeterministic-v4'
        self.agent.num_steps_between_copying_online_weights_to_target = 10000
        self.learning_rate = 0.00025
        self.agent.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.exploration.evaluation_policy = 'EGreedy'
        self.exploration.evaluation_epsilon = 0.05
        self.num_heatup_steps = 50000
        self.evaluation_episodes = 1
        self.evaluate_every_x_episodes = 100


class Breakout_C51(Preset):
    def __init__(self):
        Preset.__init__(self, CategoricalDQN, Atari, ExplorationParameters)
        self.env.level = 'BreakoutDeterministic-v4'
        self.agent.num_steps_between_copying_online_weights_to_target = 10000
        self.learning_rate = 0.00025
        self.agent.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.01
        self.exploration.epsilon_decay_steps = 1000000
        self.env.reward_clipping_max = 1.0
        self.env.reward_clipping_min = -1.0
        self.exploration.evaluation_policy = 'EGreedy'
        self.exploration.evaluation_epsilon = 0.001
        self.num_heatup_steps = 50000
        self.evaluation_episodes = 1
        self.evaluate_every_x_episodes = 5000000



class Breakout_QRDQN(Preset):
    def __init__(self):
        Preset.__init__(self, QuantileRegressionDQN, Atari, ExplorationParameters)
        self.env.level = 'BreakoutDeterministic-v4'
        self.agent.num_steps_between_copying_online_weights_to_target = 10000
        self.learning_rate = 0.00025
        self.agent.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.01
        self.exploration.epsilon_decay_steps = 1000000
        self.exploration.evaluation_policy = 'EGreedy'
        self.exploration.evaluation_epsilon = 0.001
        self.num_heatup_steps = 50000
        self.evaluation_episodes = 1
        self.evaluate_every_x_episodes = 50


class Atari_DQN_TestBench(Preset):
    def __init__(self):
        Preset.__init__(self, DQN, Atari, ExplorationParameters)
        self.env.level = 'BreakoutDeterministic-v4'
        self.agent.num_steps_between_copying_online_weights_to_target = 10000
        self.learning_rate = 0.00025
        self.agent.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.exploration.evaluation_policy = 'EGreedy'
        self.exploration.evaluation_epsilon = 0.05
        self.num_heatup_steps = 10000
        self.evaluation_episodes = 25
        self.evaluate_every_x_episodes = 1000
        self.num_training_iterations = 500


class Doom_Basic_PG(Preset):
    def __init__(self):
        Preset.__init__(self, PolicyGradient, Doom, CategoricalExploration)
        self.env.level = 'basic'
        self.agent.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.00001
        self.num_heatup_steps = 0
        self.agent.beta_entropy = 0.01


class InvertedPendulum_PG(Preset):
    def __init__(self):
        Preset.__init__(self, PolicyGradient, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'InvertedPendulum-v1'
        self.agent.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0


class Pendulum_PG(Preset):
    def __init__(self):
        Preset.__init__(self, PolicyGradient, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'Pendulum-v0'
        self.agent.policy_gradient_rescaler = 'FUTURE_RETURN_NORMALIZED_BY_TIMESTEP'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.agent.apply_gradients_every_x_episodes = 10


class Pendulum_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'Pendulum-v0'
        self.learning_rate = 0.001
        self.num_heatup_steps = 1000
        self.env.normalize_observation = False

        self.test = True
        self.test_max_step_threshold = 100
        self.test_min_return_threshold = -250


class InvertedPendulum_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, GymVectorObservation, OUExploration)
        self.env.level = 'InvertedPendulum-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100
        self.env.normalize_observation = True


class InvertedPendulum_PPO(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'InvertedPendulum-v1'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 5000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.96
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.agent.shared_optimizer = False
        self.agent.async_training = True
        self.env.normalize_observation = True


class Pendulum_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Pendulum-v0'
        self.learning_rate = 0.00005
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True
        self.agent.beta_entropy = 0.01


class Hopper_DPPO(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.00001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 5000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.96
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.agent.async_training = True
        self.env.normalize_observation = True


class InvertedPendulum_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'InvertedPendulum-v1'
        self.learning_rate = 0.00005
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True

class Humanoid_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Humanoid-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Hopper_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class InvertedPendulum_ClippedPPO_Roboschool(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, Roboschool, ExplorationParameters)
        self.env.level = 'RoboschoolInvertedPendulum-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class HalfCheetah_ClippedPPO_Roboschool(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, Roboschool, ExplorationParameters)
        self.env.level = 'RoboschoolHalfCheetah-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Hopper_ClippedPPO_Roboschool(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, Roboschool, ExplorationParameters)
        self.env.level = 'RoboschoolHopper-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Ant_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Ant-v1'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Hopper_ClippedPPO_Distributed(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.00001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 10000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'LBFGS'
        self.env.normalize_observation = True


class Hopper_DDPG_Roboschool(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, Roboschool, OUExploration)
        self.env.level = 'RoboschoolHopper-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100


class Hopper_PPO_Roboschool(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, Roboschool, ExplorationParameters)
        self.env.level = 'RoboschoolHopper-v1'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 5000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'GENERALIZED_ADVANTAGE_ESTIMATION'
        self.agent.gae_lambda = 0.96
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'LBFGS'


class Hopper_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, GymVectorObservation, OUExploration)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100
        self.env.normalize_observation = True


class Hopper_DDDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDDPG, GymVectorObservation, OUExploration)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 100
        self.env.normalize_observation = True


class Hopper_PPO(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 5000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.96
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'LBFGS'
        # self.clip_gradients = 2
        self.env.normalize_observation = True


class Walker_PPO(Preset):
    def __init__(self):
        Preset.__init__(self, PPO, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'Walker2d-v1'
        self.learning_rate = 0.001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 5000
        self.agent.discount = 0.99
        self.batch_size = 128
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.gae_lambda = 0.96
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'LBFGS'
        self.env.normalize_observation = True


class HalfCheetah_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, GymVectorObservation, OUExploration)
        self.env.level = 'HalfCheetah-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.env.normalize_observation = True


class Ant_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, GymVectorObservation, OUExploration)
        self.env.level = 'Ant-v1'
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.env.normalize_observation = True


class Pendulum_NAF(Preset):
    def __init__(self):
        Preset.__init__(self, NAF, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'Pendulum-v0'
        self.learning_rate = 0.001
        self.num_heatup_steps = 1000
        self.batch_size = 100
        # self.env.reward_scaling = 1000

        self.test = True
        self.test_max_step_threshold = 100
        self.test_min_return_threshold = -250


class InvertedPendulum_NAF(Preset):
    def __init__(self):
        Preset.__init__(self, NAF, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'InvertedPendulum-v1'
        self.learning_rate = 0.001
        self.num_heatup_steps = 1000
        self.batch_size = 100


class Hopper_NAF(Preset):
    def __init__(self):
        Preset.__init__(self, NAF, GymVectorObservation, AdditiveNoiseExploration)
        self.env.level = 'Hopper-v1'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 1000
        self.batch_size = 100
        self.agent.async_training = True
        self.env.normalize_observation = True


class CartPole_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, GymVectorObservation, ExplorationParameters)
        self.env.level = 'CartPole-v0'
        self.learning_rate = 0.00025
        self.agent.num_episodes_in_experience_replay = 200
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 1000
        self.exploration.final_epsilon = 0.1
        self.agent.discount = 1.0

        self.test = True
        self.test_max_step_threshold = 200
        self.test_min_return_threshold = 150


class Doom_Basic_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.agent.num_playing_steps_between_two_training_steps = 1


class Montezuma_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Atari, ExplorationParameters)
        self.env.level = 'MontezumaRevenge-v0'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.agent.num_playing_steps_between_two_training_steps = 1


class Breakout_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Atari, ExplorationParameters)
        self.env.level = 'BreakoutDeterministic-v4'
        self.learning_rate = 0.00025
        self.agent.num_transitions_in_experience_replay = 1000000
        self.exploration.initial_epsilon = 1.0
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.num_heatup_steps = 50000


class Doom_Health_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Doom, ExplorationParameters)
        self.env.level = 'HEALTH_GATHERING'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 10000
        self.agent.num_playing_steps_between_two_training_steps = 1


class Doom_Health_DQN(Preset):
    def __init__(self):
        Preset.__init__(self, DQN, Doom, ExplorationParameters)
        self.env.level = 'HEALTH_GATHERING'
        self.agent.num_episodes_in_experience_replay = 200
        self.learning_rate = 0.00025
        self.num_heatup_steps = 1000
        self.exploration.epsilon_decay_steps = 10000
        self.agent.num_steps_between_copying_online_weights_to_target = 1000


class Pong_NEC_LSTM(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Atari, ExplorationParameters)
        self.env.level = 'PongDeterministic-v4'
        self.learning_rate = 0.001
        self.agent.num_transitions_in_experience_replay = 1000000
        self.agent.middleware_type = MiddlewareTypes.LSTM
        self.exploration.initial_epsilon = 0.5
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.num_heatup_steps = 500


class Pong_NEC(Preset):
    def __init__(self):
        Preset.__init__(self, NEC, Atari, ExplorationParameters)
        self.env.level = 'PongDeterministic-v4'
        self.learning_rate = 0.001
        self.agent.num_transitions_in_experience_replay = 100000
        self.exploration.initial_epsilon = 0.5
        self.exploration.final_epsilon = 0.1
        self.exploration.epsilon_decay_steps = 1000000
        self.num_heatup_steps = 50000


class CartPole_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, CategoricalExploration)
        self.env.level = 'CartPole-v0'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 200.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.01
        self.agent.num_steps_between_gradient_updates = 5
        self.agent.middleware_type = MiddlewareTypes.FC

        self.test = True
        self.test_max_step_threshold = 1000
        self.test_min_return_threshold = 150
        self.test_num_workers = 8


class MountainCar_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, CategoricalExploration)
        self.env.level = 'MountainCar-v0'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 200.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.01
        self.agent.num_steps_between_gradient_updates = 5
        self.agent.middleware_type = MiddlewareTypes.FC


class InvertedPendulum_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'InvertedPendulum-v1'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 200.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 30
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.005
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC


class Hopper_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'Hopper-v1'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        self.env.reward_scaling = 20.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 0.98
        self.agent.beta_entropy = 0.005
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC


class HopperIceWall_A3C(Hopper_A3C):
    def __init__(self):
        Hopper_A3C.__init__(self)
        self.env.level = 'HopperIceWall-v0'


class HopperStairs_A3C(Hopper_A3C):
    def __init__(self):
        Hopper_A3C.__init__(self)
        self.env.level = 'HopperStairs-v0'


class HopperBullet_A3C(Hopper_A3C):
    def __init__(self):
        Hopper_A3C.__init__(self)
        self.env.level = 'HopperBulletEnv-v0'


class Kuka_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'KukaBulletEnv-v0'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Minitaur_ClippedPPO(Preset):
    def __init__(self):
        Preset.__init__(self, ClippedPPO, GymVectorObservation, ExplorationParameters)
        self.env.level = 'MinitaurBulletEnv-v0'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.num_consecutive_training_steps = 1
        self.agent.num_consecutive_playing_steps = 2048
        self.agent.discount = 0.99
        self.batch_size = 64
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.gae_lambda = 0.95
        self.visualization.dump_csv = True
        self.agent.optimizer_type = 'Adam'
        self.env.normalize_observation = True


class Walker_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'Walker2d-v1'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        self.env.reward_scaling = 20.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.005
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC


class Ant_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'Ant-v1'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        self.env.reward_scaling = 20.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.005
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC
        self.env.normalize_observation = True


class AntBullet_A3C(Ant_A3C):
    def __init__(self):
        Ant_A3C.__init__(self)
        self.env.level = 'AntBulletEnv-v0'


class AntMaze_A3C(Ant_A3C):
    def __init__(self):
        Ant_A3C.__init__(self)
        self.env.level = 'AntMaze-v0'


class Humanoid_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'Humanoid-v1'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        self.env.reward_scaling = 20.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.005
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC
        self.env.normalize_observation = True


class Pendulum_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'Pendulum-v0'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.agent.optimizer_type = 'Adam'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.agent.discount = 0.99
        self.agent.num_steps_between_gradient_updates = 5
        self.agent.gae_lambda = 1



class BipedalWalker_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, GymVectorObservation, EntropyExploration)
        self.env.level = 'BipedalWalker-v2'
        self.agent.policy_gradient_rescaler = 'A_VALUE'
        self.agent.optimizer_type = 'RMSProp'
        self.learning_rate = 0.00002
        self.num_heatup_steps = 0
        self.env.reward_scaling = 50.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 10
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.005
        self.clip_gradients = None
        self.agent.middleware_type = MiddlewareTypes.FC


class Doom_Basic_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, Doom, CategoricalExploration)
        self.env.level = 'basic'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 100.
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 30
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.01
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC


class Pong_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, Atari, CategoricalExploration)
        self.env.level = 'PongDeterministic-v4'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        self.env.reward_scaling = 1.
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 1.
        self.agent.beta_entropy = 0.01
        self.clip_gradients = 40.0
        self.agent.middleware_type = MiddlewareTypes.FC


class Breakout_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, Atari, CategoricalExploration)
        self.env.level = 'BreakoutDeterministic-v4'
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 200
        self.env.reward_scaling = 1.
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 20
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.05
        self.clip_gradients = 40.0
        self.agent.middleware_type = MiddlewareTypes.FC


class Carla_A3C(Preset):
    def __init__(self):
        Preset.__init__(self, ActorCritic, Carla, EntropyExploration)
        self.agent.embedder_complexity = EmbedderComplexity.Deep
        self.agent.policy_gradient_rescaler = 'GAE'
        self.learning_rate = 0.0001
        self.num_heatup_steps = 0
        # self.env.reward_scaling = 1.0e9
        self.agent.discount = 0.99
        self.agent.apply_gradients_every_x_episodes = 1
        self.agent.num_steps_between_gradient_updates = 30
        self.agent.gae_lambda = 1
        self.agent.beta_entropy = 0.01
        self.clip_gradients = 40
        self.agent.middleware_type = MiddlewareTypes.FC


class Carla_DDPG(Preset):
    def __init__(self):
        Preset.__init__(self, DDPG, Carla, OUExploration)
        self.agent.embedder_complexity = EmbedderComplexity.Deep
        self.learning_rate = 0.0001
        self.num_heatup_steps = 1000
        self.agent.num_consecutive_training_steps = 5


class Carla_BC(Preset):
    def __init__(self):
        Preset.__init__(self, BC, Carla, ExplorationParameters)
        self.agent.embedder_complexity = EmbedderComplexity.Deep
        self.agent.load_memory_from_file_path = 'datasets/carla_town1.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 5000


class Doom_Basic_BC(Preset):
    def __init__(self):
        Preset.__init__(self, BC, Doom, ExplorationParameters)
        self.env.level = 'basic'
        self.agent.load_memory_from_file_path = 'datasets/doom_basic.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 100
        self.num_training_iterations = 2000


class Doom_Defend_BC(Preset):
    def __init__(self):
        Preset.__init__(self, BC, Doom, ExplorationParameters)
        self.env.level = 'defend'
        self.agent.load_memory_from_file_path = 'datasets/doom_defend.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 100


class Doom_Deathmatch_BC(Preset):
    def __init__(self):
        Preset.__init__(self, BC, Doom, ExplorationParameters)
        self.env.level = 'deathmatch'
        self.agent.load_memory_from_file_path = 'datasets/doom_deathmatch.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 100


class MontezumaRevenge_BC(Preset):
    def __init__(self):
        Preset.__init__(self, BC, Atari, ExplorationParameters)
        self.env.level = 'MontezumaRevenge-v0'
        self.agent.load_memory_from_file_path = 'datasets/montezuma_revenge.p'
        self.learning_rate = 0.0005
        self.num_heatup_steps = 0
        self.evaluation_episodes = 5
        self.batch_size = 120
        self.evaluate_every_x_training_iterations = 100
        self.exploration.evaluation_epsilon = 0.05
        self.exploration.evaluation_policy = 'EGreedy'
        self.env.frame_skip = 1
