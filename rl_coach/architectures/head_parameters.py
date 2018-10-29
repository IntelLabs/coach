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

from typing import Type

from rl_coach.base_parameters import NetworkComponentParameters


class HeadParameters(NetworkComponentParameters):
    def __init__(self, parameterized_class_name: str, activation_function: str = 'relu', name: str= 'head',
                 num_output_head_copies: int=1, rescale_gradient_from_head_by_factor: float=1.0,
                 loss_weight: float=1.0, dense_layer=None):
        super().__init__(dense_layer=dense_layer)
        self.activation_function = activation_function
        self.name = name
        self.num_output_head_copies = num_output_head_copies
        self.rescale_gradient_from_head_by_factor = rescale_gradient_from_head_by_factor
        self.loss_weight = loss_weight
        self.parameterized_class_name = parameterized_class_name


class PPOHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='ppo_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="PPOHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class VHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='v_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="VHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class CategoricalQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='categorical_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="CategoricalQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class RegressionHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None, scheme=None):
        super().__init__(parameterized_class_name="RegressionHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class DDPGActorHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='policy_head_params', batchnorm: bool=True,
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="DDPGActor", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)
        self.batchnorm = batchnorm


class DNDQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='dnd_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="DNDQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class DuelingQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='dueling_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="DuelingQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class MeasurementsPredictionHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='measurements_prediction_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="MeasurementsPredictionHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class NAFHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='naf_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="NAFHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class PolicyHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='tanh', name: str='policy_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="PolicyHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class PPOVHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='ppo_v_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="PPOVHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class QHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="QHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class QuantileRegressionQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='quantile_regression_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="QuantileRegressionQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)


class RainbowQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='rainbow_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None):
        super().__init__(parameterized_class_name="RainbowQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)
