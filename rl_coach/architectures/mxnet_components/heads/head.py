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


from typing import Dict, List, Union, Tuple

import mxnet as mx
from mxnet.initializer import Initializer, register
from mxnet.gluon import nn, loss
from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


LOSS_OUT_TYPE_LOSS = 'loss'
LOSS_OUT_TYPE_REGULARIZATION = 'regularization'


@register
class NormalizedRSSInitializer(Initializer):
    """
    Standardizes Root Sum of Squares along the input channel dimension.
    Used for Dense layer weight matrices only (ie. do not use on Convolution kernels).
    MXNet Dense layer weight matrix is of shape (out_ch, in_ch), so standardize across axis 1.
    Root Sum of Squares set to `rss`, which is 1.0 by default.
    Called `normalized_columns_initializer` in TensorFlow backend (but we work with rows instead of columns for MXNet).
    """
    def __init__(self, rss=1.0):
        super(NormalizedRSSInitializer, self).__init__(rss=rss)
        self.rss = float(rss)

    def _init_weight(self, name, arr):
        mx.nd.random.normal(0, 1, out=arr)
        sample_rss = arr.square().sum(axis=1).sqrt()
        scalers = self.rss / sample_rss
        arr *= scalers.expand_dims(1)


class LossInputSchema(object):
    """
    Helper class to contain schema for loss hybrid_forward input
    """
    def __init__(self, head_outputs: List[str], agent_inputs: List[str], targets: List[str]):
        """
        :param head_outputs: list of argument names in hybrid_forward that are outputs of the head.
            The order and number MUST MATCH the output from the head.
        :param agent_inputs: list of argument names in hybrid_forward that are inputs from the agent.
            The order and number MUST MATCH `output_<head_type_idx>_<order>` for this head.
        :param targets: list of argument names in hybrid_forward that are targets for the loss.
            The order and number MUST MATCH targets passed from the agent.
        """
        self._head_outputs = head_outputs
        self._agent_inputs = agent_inputs
        self._targets = targets

    @property
    def head_outputs(self):
        return self._head_outputs

    @property
    def agent_inputs(self):
        return self._agent_inputs

    @property
    def targets(self):
        return self._targets


class HeadLoss(loss.Loss):
    """
    ABC for loss functions of each head. Child class must implement input_schema() and loss_forward()
    """
    def __init__(self, *args, **kwargs):
        super(HeadLoss, self).__init__(*args, **kwargs)
        self._output_schema = None  # type: List[str]

    @property
    def input_schema(self) -> LossInputSchema:
        """
        :return: schema for input of hybrid_forward. Read docstring for LossInputSchema for details.
        """
        raise NotImplementedError

    @property
    def output_schema(self) -> List[str]:
        """
        :return: schema for output of hybrid_forward. Must contain 'loss' and 'regularization' keys at least once.
            The order and total number must match that of returned values from the loss. 'loss' and 'regularization'
            are special keys. Any other string is treated as auxiliary outputs and must include match auxiliary
            fetch names returned by the head.
        """
        return self._output_schema

    def forward(self, *args):
        """
        Override forward() so that number of outputs can be checked against the schema
        """
        outputs = super(HeadLoss, self).forward(*args)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, NDArray) or isinstance(outputs, Symbol)
            num_outputs = 1
        assert num_outputs == len(self.output_schema), "Number of outputs don't match schema ({} != {})".format(
            num_outputs, len(self.output_schema))
        return outputs

    def _loss_output(self, outputs: List[Tuple[Union[NDArray, Symbol], str]]):
        """
        Must be called on the output from hybrid_forward().
        Saves the returned output as the schema and returns output values in a list
        :return: list of output values
        """
        output_schema = [o[1] for o in outputs]
        assert self._output_schema is None or self._output_schema == output_schema
        self._output_schema = output_schema
        return tuple(o[0] for o in outputs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        Passes the cal to loss_forward() and constructs output schema from its output by calling loss_output()
        """
        return self._loss_output(self.loss_forward(F, x, *args, **kwargs))

    def loss_forward(self, F, x, *args, **kwargs) -> List[Tuple[Union[NDArray, Symbol], str]]:
        """
        Similar to hybrid_forward, but returns list of (NDArray, type_str)
        """
        raise NotImplementedError


class Head(nn.HybridBlock):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition,
                 network_name: str, head_type_idx: int=0, loss_weight: float=1., is_local: bool=True,
                 activation_function: str='relu', dense_layer: None=None):
        """
        A head is the final part of the network. It takes the embedding from the middleware embedder and passes it
        through a neural network to produce the output of the network. There can be multiple heads in a network, and
        each one has an assigned loss function. The heads are algorithm dependent.

        :param agent_parameters: containing algorithm parameters such as clip_likelihood_ratio_using_epsilon
            and beta_entropy.
        :param spaces: containing action spaces used for defining size of network output.
        :param network_name: name of head network. currently unused.
        :param head_type_idx: index of head network. currently unused.
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param is_local: flag to denote if network is local. currently unused.
        :param activation_function: activation function to use between layers. currently unused.
        :param dense_layer: type of dense layer to use in network. currently unused.
        """
        super(Head, self).__init__()
        self.head_type_idx = head_type_idx
        self.network_name = network_name
        self.loss_weight = loss_weight
        self.is_local = is_local
        self.ap = agent_parameters
        self.spaces = spaces
        self.return_type = None
        self.activation_function = activation_function
        self.dense_layer = dense_layer
        self._num_outputs = None

    def loss(self) -> HeadLoss:
        """
        Returns loss block to be used for specific head implementation.

        :return: loss block (can be called as function) for outputs returned by the head network.
        """
        raise NotImplementedError()

    @property
    def num_outputs(self):
        """ Returns number of outputs that forward() call will return

        :return:
        """
        assert self._num_outputs is not None, 'must call forward() once to configure number of outputs'
        return self._num_outputs

    def forward(self, *args):
        """
        Override forward() so that number of outputs can be automatically set
        """
        outputs = super(Head, self).forward(*args)
        if isinstance(outputs, tuple):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, NDArray) or isinstance(outputs, Symbol)
            num_outputs = 1
        if self._num_outputs is None:
            self._num_outputs = num_outputs
        else:
            assert self._num_outputs == num_outputs, 'Number of outputs cannot change ({} != {})'.format(
                self._num_outputs, num_outputs)
        assert self._num_outputs == len(self.loss().input_schema.head_outputs)
        return outputs

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        Used for forward pass through head network.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param x: middleware state representation, of shape (batch_size, in_channels).
        :return: final output of network, that will be used in loss calculations.
        """
        raise NotImplementedError()
