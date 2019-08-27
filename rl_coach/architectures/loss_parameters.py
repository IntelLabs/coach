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

from rl_coach.base_parameters import Parameters


class LossParameters(Parameters):
    def __init__(self, parameterized_class_name: str,
                 name: str= 'loss',
                 num_output_head_copies: int=1,
                 loss_weight: float=1.0,
                 is_training=False):
        super().__init__()
        self.name = name
        self.num_output_head_copies = num_output_head_copies
        self.loss_weight = loss_weight
        self.parameterized_class_name = parameterized_class_name
        self.is_training = is_training

    @property
    def path(self):
        return 'rl_coach.architectures.tensorflow_components.heads:' + self.parameterized_class_name



class QLossParameters(LossParameters):
    def __init__(self, name: str='q_loss_params',
                 num_output_head_copies: int = 1,
                 loss_weight: float = 1.0):
        super().__init__(parameterized_class_name="QLoss",
                         name=name,
                         num_output_head_copies=num_output_head_copies,
                         loss_weight=loss_weight)