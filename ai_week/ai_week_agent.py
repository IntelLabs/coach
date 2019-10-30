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

from typing import Union

import numpy as np

from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent, PolicyGradientRescaler
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters

from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.memories.episodic.single_episode_buffer import SingleEpisodeBufferParameters
from rl_coach.spaces import DiscreteActionSpace, BoxActionSpace




class WorkShopHeadParameters(HeadParameters):
    def __init__(self):
        super().__init__(parameterized_class_name="AiWeekHead")

    @property
    def path(self):
        return 'ai_week.ai_week_head:AiWeekHead'


class NetwokTopology(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [WorkShopHeadParameters()]


class SimplePGAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_gradient_updates: (int)
        The number of steps between calculating gradients for the collected data. In the A3C paper, this parameter is
        called t_max. Since this algorithm is on-policy, only the steps collected between each two gradient calculations
        are used in the batch.
    """
    def __init__(self):
        super().__init__()
        self.num_steps_between_gradient_updates = 20000  # this is called t_max in all the papers


class AiWeekAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=SimplePGAlgorithmParameters(),
                         exploration={DiscreteActionSpace: CategoricalParameters(),
                                      BoxActionSpace: AdditiveNoiseParameters()},
                         memory=SingleEpisodeBufferParameters(),
                         networks={"main": NetwokTopology()})

    @property
    def path(self):
        return 'ai_week.ai_week_agent:SimplePgAgent'


class SimplePgAgent(PolicyOptimizationAgent):
    def __init__(self, agent_parameters):
        super().__init__(agent_parameters)
        # self.returns_mean = self.register_signal('Returns Mean')
        # self.returns_variance = self.register_signal('Returns Variance')

    def learn_from_batch(self, batch):
        # batch contains a list of episodes to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # FUTURE_RETURN
        total_returns = batch.n_step_discounted_rewards()

        actions = batch.actions()
        self.agent_logger.create_signal_value('Returns Mean', np.mean(total_returns))
        self.agent_logger.create_signal_value('Returns Variance', np.std(total_returns))
        # self.returns_mean.add_sample(np.mean(total_returns))
        # self.returns_variance.add_sample(np.std(total_returns))

        # Both the inputs and the actions are inputs to the loss head
        inputs_dict = batch.states(['observation'])
        inputs_dict.update({'output_0_0': actions})

        result = self.networks['main'].online_network.accumulate_gradients(inputs=inputs_dict, targets=total_returns)

        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
