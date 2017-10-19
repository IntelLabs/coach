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

import ngraph as ng
import ngraph.frontends.neon as neon
from ngraph.util.names import name_scope
import numpy as np


def mean_squared_error(targets, outputs, weights=1.0, scope=""):
    with name_scope(scope):
        # TODO: reduce mean over the action axis
        loss = ng.squared_L2(targets - outputs)
        weighted_loss = loss * weights
        return weighted_loss
