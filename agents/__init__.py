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
from agents.actor_critic_agent import ActorCriticAgent
from agents.agent import Agent
from agents.bc_agent import BCAgent
from agents.bootstrapped_dqn_agent import BootstrappedDQNAgent
from agents.categorical_dqn_agent import CategoricalDQNAgent
from agents.clipped_ppo_agent import ClippedPPOAgent
from agents.ddpg_agent import DDPGAgent
from agents.ddqn_agent import DDQNAgent
from agents.dfp_agent import DFPAgent
from agents.dqn_agent import DQNAgent
from agents.human_agent import HumanAgent
from agents.imitation_agent import ImitationAgent
from agents.mmc_agent import MixedMonteCarloAgent
from agents.n_step_q_agent import NStepQAgent
from agents.naf_agent import NAFAgent
from agents.nec_agent import NECAgent
from agents.pal_agent import PALAgent
from agents.policy_gradients_agent import PolicyGradientsAgent
from agents.policy_optimization_agent import PolicyOptimizationAgent
from agents.ppo_agent import PPOAgent
from agents.qr_dqn_agent import QuantileRegressionDQNAgent
from agents.value_optimization_agent import ValueOptimizationAgent

__all__ = [ActorCriticAgent,
           Agent,
           BCAgent,
           BootstrappedDQNAgent,
           CategoricalDQNAgent,
           ClippedPPOAgent,
           DDPGAgent,
           DDQNAgent,
           DFPAgent,
           DQNAgent,
           HumanAgent,
           ImitationAgent,
           MixedMonteCarloAgent,
           NAFAgent,
           NECAgent,
           NStepQAgent,
           PALAgent,
           PPOAgent,
           PolicyGradientsAgent,
           PolicyOptimizationAgent,
           QuantileRegressionDQNAgent,
           ValueOptimizationAgent]
