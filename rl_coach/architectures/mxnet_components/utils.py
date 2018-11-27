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


"""
Module defining utility functions
"""
import inspect
from typing import Any, Dict, Generator, Iterable, List, Tuple, Union
from types import ModuleType

import mxnet as mx
from mxnet import gluon, nd
from mxnet.ndarray import NDArray
import numpy as np

from rl_coach.core_types import GradientClippingMethod

nd_sym_type = Union[mx.nd.NDArray, mx.sym.Symbol]


def to_mx_ndarray(data: Union[list, tuple, np.ndarray, NDArray, int, float], ctx: mx.Context=None) ->\
        Union[List[NDArray], Tuple[NDArray], NDArray]:
    """
    Convert data to mx.nd.NDArray. Data can be a list or tuple of np.ndarray, int, or float or
    it can be np.ndarray, int, or float
    :param data: input data to be converted
    :param ctx: context of the data (CPU, GPU0, GPU1, etc.)
    :return: converted output data
    """
    if isinstance(data, list):
        data = [to_mx_ndarray(d, ctx=ctx) for d in data]
    elif isinstance(data, tuple):
        data = tuple(to_mx_ndarray(d, ctx=ctx) for d in data)
    elif isinstance(data, np.ndarray):
        data = nd.array(data, ctx=ctx)
    elif isinstance(data, NDArray):
        assert data.context == ctx
        pass
    elif isinstance(data, int) or isinstance(data, float):
        data = nd.array([data], ctx=ctx)
    else:
        raise TypeError('Unsupported data type: {}'.format(type(data)))
    return data


def asnumpy_or_asscalar(data: Union[NDArray, list, tuple]) -> Union[np.ndarray, np.number, list, tuple]:
    """
    Convert NDArray (or list or tuple of NDArray) to numpy. If shape is (1,), then convert to scalar instead.
    NOTE: This behavior is consistent with tensorflow
    :param data: NDArray or list or tuple of NDArray
    :return: data converted to numpy ndarray or to numpy scalar
    """
    if isinstance(data, list):
        data = [asnumpy_or_asscalar(d) for d in data]
    elif isinstance(data, tuple):
        data = tuple(asnumpy_or_asscalar(d) for d in data)
    elif isinstance(data, NDArray):
        data = data.asscalar() if data.shape == (1,) else data.asnumpy()
    else:
        raise TypeError('Unsupported data type: {}'.format(type(data)))
    return data


def global_norm(arrays: Union[Generator[NDArray, NDArray, NDArray], List[NDArray], Tuple[NDArray]]) -> NDArray:
    """
    Calculate global norm on list or tuple of NDArrays using this formula:
        `global_norm = sqrt(sum([l2norm(p)**2 for p in parameters]))`

    :param arrays: list or tuple of parameters to calculate global norm on
    :return: single-value NDArray
    """
    def _norm(array):
        if array.stype == 'default':
            x = array.reshape((-1,))
            return nd.dot(x, x)
        return array.norm().square()

    total_norm = nd.add_n(*[_norm(arr) for arr in arrays])
    total_norm = nd.sqrt(total_norm)
    return total_norm


def split_outputs_per_head(outputs: Tuple[NDArray], heads: list) -> List[List[NDArray]]:
    """
    Split outputs into outputs per head
    :param outputs: list of all outputs
    :param heads: list of all heads
    :return: list of outputs for each head
    """
    head_outputs = []
    for h in heads:
        head_outputs.append(list(outputs[:h.num_outputs]))
        outputs = outputs[h.num_outputs:]
    assert len(outputs) == 0
    return head_outputs


def split_targets_per_loss(targets: list, losses: list) -> List[list]:
    """
    Splits targets into targets per loss
    :param targets: list of all targets (typically numpy ndarray)
    :param losses: list of all losses
    :return: list of targets for each loss
    """
    loss_targets = list()
    for l in losses:
        loss_data_len = len(l.input_schema.targets)
        assert len(targets) >= loss_data_len, "Data length doesn't match schema"
        loss_targets.append(targets[:loss_data_len])
        targets = targets[loss_data_len:]
    assert len(targets) == 0
    return loss_targets


def get_loss_agent_inputs(inputs: Dict[str, np.ndarray], head_type_idx: int, loss: Any) -> List[np.ndarray]:
    """
    Collects all inputs with prefix 'output_<head_idx>_' and matches them against agent_inputs in loss input schema.
    :param inputs: list of all agent inputs
    :param head_type_idx: head-type index of the corresponding head
    :param loss: corresponding loss
    :return: list of agent inputs for this loss. This list matches the length in loss input schema.
    """
    loss_inputs = list()
    for k in sorted(inputs.keys()):
        if k.startswith('output_{}_'.format(head_type_idx)):
            loss_inputs.append(inputs[k])
    # Enforce that number of inputs for head_type are the same as agent_inputs specified by loss input_schema
    assert len(loss_inputs) == len(loss.input_schema.agent_inputs), "agent_input length doesn't match schema"
    return loss_inputs


def align_loss_args(
        head_outputs: List[NDArray],
        agent_inputs: List[np.ndarray],
        targets: List[np.ndarray],
        loss: Any) -> List[np.ndarray]:
    """
    Creates a list of arguments from head_outputs, agent_inputs, and targets aligned with parameters of
    loss.loss_forward() based on their name in loss input_schema
    :param head_outputs: list of all head_outputs for this loss
    :param agent_inputs: list of all agent_inputs for this loss
    :param targets: list of all targets for this loss
    :param loss: corresponding loss
    :return: list of arguments in correct order to be passed to loss
    """
    arg_list = list()
    schema = loss.input_schema
    assert len(schema.head_outputs) == len(head_outputs)
    assert len(schema.agent_inputs) == len(agent_inputs)
    assert len(schema.targets) == len(targets)

    prev_found = True
    for arg_name in inspect.getfullargspec(loss.loss_forward).args[2:]:  # First two args are self and F
        found = False
        for schema_list, data in [(schema.head_outputs, head_outputs),
                                  (schema.agent_inputs, agent_inputs),
                                  (schema.targets, targets)]:
            try:
                arg_list.append(data[schema_list.index(arg_name)])
                found = True
                break
            except ValueError:
                continue
        assert not found or prev_found, "missing arguments detected!"
        prev_found = found
    return arg_list


def to_tuple(data: Union[tuple, list, Any]):
    """
    If input is list, it is converted to tuple. If it's tuple, it is returned untouched. Otherwise
    returns a single-element tuple of the data.
    :return: tuple-ified data
    """
    if isinstance(data, tuple):
        pass
    elif isinstance(data, list):
        data = tuple(data)
    else:
        data = (data,)
    return data


def to_list(data: Union[tuple, list, Any]):
    """
    If input is tuple, it is converted to list. If it's list, it is returned untouched. Otherwise
    returns a single-element list of the data.
    :return: list-ified data
    """
    if isinstance(data, list):
        pass
    elif isinstance(data, tuple):
        data = list(data)
    else:
        data = [data]
    return data


def loss_output_dict(output: List[NDArray], schema: List[str]) -> Dict[str, List[NDArray]]:
    """
    Creates a dictionary for loss output based on the output schema. If two output values have the same
    type string in the schema they are concatenated in the same dicrionary item.
    :param output: list of output values
    :param schema: list of type-strings for output values
    :return: dictionary of keyword to list of NDArrays
    """
    assert len(output) == len(schema)
    output_dict = dict()
    for name, val in zip(schema, output):
        if name in output_dict:
            output_dict[name].append(val)
        else:
            output_dict[name] = [val]
    return output_dict


def clip_grad(
        grads: Union[Generator[NDArray, NDArray, NDArray], List[NDArray], Tuple[NDArray]],
        clip_method: GradientClippingMethod,
        clip_val: float,
        inplace=True) -> List[NDArray]:
    """
    Clip gradient values inplace
    :param grads: gradients to be clipped
    :param clip_method: clipping method
    :param clip_val: clipping value. Interpreted differently depending on clipping method.
    :param inplace: modify grads if True, otherwise create NDArrays
    :return: clipped gradients
    """
    output = list(grads) if inplace else list(nd.empty(g.shape) for g in grads)
    if clip_method == GradientClippingMethod.ClipByGlobalNorm:
        norm_unclipped_grads = global_norm(grads)
        scale = clip_val / (norm_unclipped_grads.asscalar() + 1e-8)  # todo: use branching operators?
        if scale < 1.0:
            for g, o in zip(grads, output):
                nd.broadcast_mul(g, nd.array([scale]), out=o)
    elif clip_method == GradientClippingMethod.ClipByValue:
        for g, o in zip(grads, output):
            g.clip(-clip_val, clip_val, out=o)
    elif clip_method == GradientClippingMethod.ClipByNorm:
        for g, o in zip(grads, output):
            nd.broadcast_mul(g, nd.minimum(1.0, clip_val / (g.norm() + 1e-8)), out=o)
    else:
        raise KeyError('Unsupported gradient clipping method')
    return output


def hybrid_clip(F: ModuleType, x: nd_sym_type, clip_lower: nd_sym_type, clip_upper: nd_sym_type) -> nd_sym_type:
    """
    Apply clipping to input x between clip_lower and clip_upper.
    Added because F.clip doesn't support clipping bounds that are mx.nd.NDArray or mx.sym.Symbol.

    :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
    :param x: input data
    :param clip_lower: lower bound used for clipping, should be of shape (1,)
    :param clip_upper: upper bound used for clipping, should be of shape (1,)
    :return: clipped data
    """
    x_clip_lower = broadcast_like(F, clip_lower, x)
    x_clip_upper = broadcast_like(F, clip_upper, x)
    x_clipped = F.minimum(F.maximum(x, x_clip_lower), x_clip_upper)
    return x_clipped


def broadcast_like(F: ModuleType, x: nd_sym_type, y: nd_sym_type) -> nd_sym_type:
    """
    Implementation of broadcast_like using broadcast_add and broadcast_mul because ONNX doesn't support broadcast_like.
    :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
    :param x: input to be broadcast
    :param y: tensor to broadcast x like
    :return: broadcast x
    """
    return F.broadcast_mul(x, (y * 0) + 1)


def get_mxnet_activation_name(activation_name: str):
    """
    Convert coach activation name to mxnet specific activation name
    :param activation_name: name of the activation inc coach
    :return: name of the activation in mxnet
    """
    activation_functions = {
        'relu': 'relu',
        'tanh': 'tanh',
        'sigmoid': 'sigmoid',
        # FIXME Add other activations
        # 'elu': tf.nn.elu,
        'selu': 'softrelu',
        # 'leaky_relu': tf.nn.leaky_relu,
        'none': None
    }
    assert activation_name in activation_functions, \
        "Activation function must be one of the following {}. instead it was: {}".format(
            activation_functions.keys(), activation_name)
    return activation_functions[activation_name]


class OnnxHandlerBlock(object):
    """
    Helper base class for gluon blocks that must behave differently for ONNX export forward pass
    """
    def __init__(self):
        self._onnx = False

    def enable_onnx(self):
        self._onnx = True

    def disable_onnx(self):
        self._onnx = False


class ScopedOnnxEnable(object):
    """
    Helper scoped ONNX enable class
    """
    def __init__(self, net: gluon.HybridBlock):
        self._onnx_handlers = self._get_onnx_handlers(net)

    def __enter__(self):
        for b in self._onnx_handlers:
            b.enable_onnx()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for b in self._onnx_handlers:
            b.disable_onnx()

    @staticmethod
    def _get_onnx_handlers(block: gluon.HybridBlock) -> List[OnnxHandlerBlock]:
        """
        Iterates through all child blocks and return all of them that are instance of OnnxHandlerBlock
        :return: list of OnnxHandlerBlock child blocks
        """
        handlers = list()
        if isinstance(block, OnnxHandlerBlock):
            handlers.append(block)
        for child_block in block._children.values():
            handlers += ScopedOnnxEnable._get_onnx_handlers(child_block)
        return handlers
