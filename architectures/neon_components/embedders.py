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

import ngraph.frontends.neon as neon
import ngraph as ng
from ngraph.util.names import name_scope


class InputEmbedder(object):
    def __init__(self, input_size, batch_size=None, activation_function=neon.Rectlin(), name="embedder"):
        self.name = name
        self.input_size = input_size
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.weights_init = neon.GlorotInit()
        self.biases_init = neon.ConstantInit()
        self.input = None
        self.output = None

    def __call__(self, prev_input_placeholder=None):
        with name_scope(self.get_name()):
            # create the input axes
            axes = []
            if len(self.input_size) == 2:
                axis_names = ['H', 'W']
            else:
                axis_names = ['C', 'H', 'W']
            for axis_size, axis_name in zip(self.input_size, axis_names):
                axes.append(ng.make_axis(axis_size, name=axis_name))
            batch_axis_full = ng.make_axis(self.batch_size, name='N')
            input_axes = ng.make_axes(axes)

            if prev_input_placeholder is None:
                self.input = ng.placeholder(input_axes + [batch_axis_full])
            else:
                self.input = prev_input_placeholder
            self._build_module()

        return self.input, self.output(self.input)

    def _build_module(self):
        pass

    def get_name(self):
        return self.name


class ImageEmbedder(InputEmbedder):
    def __init__(self, input_size, batch_size=None, input_rescaler=255.0, activation_function=neon.Rectlin(), name="embedder"):
        InputEmbedder.__init__(self, input_size, batch_size, activation_function, name)
        self.input_rescaler = input_rescaler

    def _build_module(self):
        # image observation
        self.output = neon.Sequential([
            neon.Preprocess(functor=lambda x: x / self.input_rescaler),
            neon.Convolution((8, 8, 32), strides=4, activation=self.activation_function,
                             filter_init=self.weights_init, bias_init=self.biases_init),
            neon.Convolution((4, 4, 64), strides=2, activation=self.activation_function,
                             filter_init=self.weights_init, bias_init=self.biases_init),
            neon.Convolution((3, 3, 64), strides=1, activation=self.activation_function,
                             filter_init=self.weights_init, bias_init=self.biases_init)
        ])


class VectorEmbedder(InputEmbedder):
    def __init__(self, input_size, batch_size=None, activation_function=neon.Rectlin(), name="embedder"):
        InputEmbedder.__init__(self, input_size, batch_size, activation_function, name)

    def _build_module(self):
        # vector observation
        self.output = neon.Sequential([
                neon.Affine(nout=256, activation=self.activation_function,
                            weight_init=self.weights_init, bias_init=self.biases_init)
            ])
