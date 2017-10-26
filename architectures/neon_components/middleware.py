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


class MiddlewareEmbedder(object):
    def __init__(self, activation_function=neon.Rectlin(), name="middleware_embedder"):
        self.name = name
        self.input = None
        self.output = None
        self.weights_init = neon.GlorotInit()
        self.biases_init = neon.ConstantInit()
        self.activation_function = activation_function

    def __call__(self, input_layer):
        with name_scope(self.get_name()):
            self.input = input_layer
            self._build_module()

        return self.input, self.output(self.input)

    def _build_module(self):
        pass

    def get_name(self):
        return self.name


class FC_Embedder(MiddlewareEmbedder):
    def _build_module(self):
        self.output = neon.Sequential([
                neon.Affine(nout=512, activation=self.activation_function,
                            weight_init=self.weights_init, bias_init=self.biases_init)])
