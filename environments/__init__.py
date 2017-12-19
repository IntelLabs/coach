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

from logger import *
from utils import Enum, get_open_port
from environments.gym_environment_wrapper import *
from environments.doom_environment_wrapper import *
from environments.carla_environment_wrapper import *


class EnvTypes(Enum):
    Doom = "DoomEnvironmentWrapper"
    Gym = "GymEnvironmentWrapper"
    Carla = "CarlaEnvironmentWrapper"


def create_environment(tuning_parameters):
    env_type_name, env_type = EnvTypes().verify(tuning_parameters.env.type)
    env = eval(env_type)(tuning_parameters)
    return env



