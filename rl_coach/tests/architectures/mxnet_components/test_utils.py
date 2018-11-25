import pytest

import mxnet as mx
from mxnet import nd
import numpy as np

from rl_coach.architectures.mxnet_components.utils import *


@pytest.mark.unit_test
def test_to_mx_ndarray():
    # scalar
    assert to_mx_ndarray(1.2) == nd.array([1.2])
    # list of one scalar
    assert to_mx_ndarray([1.2]) == [nd.array([1.2])]
    # list of multiple scalars
    assert to_mx_ndarray([1.2, 3.4]) == [nd.array([1.2]), nd.array([3.4])]
    # list of lists of scalars
    assert to_mx_ndarray([[1.2], [3.4]]) == [[nd.array([1.2])], [nd.array([3.4])]]
    # numpy
    assert np.array_equal(to_mx_ndarray(np.array([[1.2], [3.4]])).asnumpy(), nd.array([[1.2], [3.4]]).asnumpy())
    # tuple
    assert to_mx_ndarray(((1.2,), (3.4,))) == ((nd.array([1.2]),), (nd.array([3.4]),))


@pytest.mark.unit_test
def test_asnumpy_or_asscalar():
    # scalar float32
    assert asnumpy_or_asscalar(nd.array([1.2])) == np.float32(1.2)
    # scalar int32
    assert asnumpy_or_asscalar(nd.array([2], dtype=np.int32)) == np.int32(2)
    # list of one scalar
    assert asnumpy_or_asscalar([nd.array([1.2])]) == [np.float32(1.2)]
    # list of multiple scalars
    assert asnumpy_or_asscalar([nd.array([1.2]), nd.array([3.4])]) == [np.float32([1.2]), np.float32([3.4])]
    # list of lists of scalars
    assert asnumpy_or_asscalar([[nd.array([1.2])], [nd.array([3.4])]]) == [[np.float32([1.2])], [np.float32([3.4])]]
    # tensor
    assert np.array_equal(asnumpy_or_asscalar(nd.array([[1.2], [3.4]])), np.array([[1.2], [3.4]], dtype=np.float32))
    # tuple
    assert (asnumpy_or_asscalar(((nd.array([1.2]),), (nd.array([3.4]),))) ==
            ((np.array([1.2], dtype=np.float32),), (np.array([3.4], dtype=np.float32),)))


@pytest.mark.unit_test
def test_global_norm():
    data = list()
    for i in range(1, 6):
        data.append(np.ones((i * 10, i * 10)) * i)
    gnorm = np.asscalar(np.sqrt(sum([np.sum(np.square(d)) for d in data])))
    assert np.isclose(gnorm, global_norm([nd.array(d) for d in data]).asscalar())


@pytest.mark.unit_test
def test_split_outputs_per_head():
    class TestHead:
        def __init__(self, num_outputs):
            self.num_outputs = num_outputs

    assert split_outputs_per_head((1, 2, 3, 4), [TestHead(2), TestHead(1), TestHead(1)]) == [[1, 2], [3], [4]]


class DummySchema:
    def __init__(self, num_head_outputs, num_agent_inputs, num_targets):
        self.head_outputs = ['head_output_{}'.format(i) for i in range(num_head_outputs)]
        self.agent_inputs = ['agent_input_{}'.format(i) for i in range(num_agent_inputs)]
        self.targets = ['target_{}'.format(i) for i in range(num_targets)]


class DummyLoss:
    def __init__(self, num_head_outputs, num_agent_inputs, num_targets):
        self.input_schema = DummySchema(num_head_outputs, num_agent_inputs, num_targets)


@pytest.mark.unit_test
def test_split_targets_per_loss():
    assert split_targets_per_loss([1, 2, 3, 4],
                                  [DummyLoss(10, 100, 2), DummyLoss(20, 200, 1), DummyLoss(30, 300, 1)]) == \
           [[1, 2], [3], [4]]


@pytest.mark.unit_test
def test_get_loss_agent_inputs():
    input_dict = {'output_0_0': [1, 2], 'output_0_1': [3, 4], 'output_1_0': [5]}
    assert get_loss_agent_inputs(input_dict, 0, DummyLoss(10, 2, 100)) == [[1, 2], [3, 4]]
    assert get_loss_agent_inputs(input_dict, 1, DummyLoss(20, 1, 200)) == [[5]]


@pytest.mark.unit_test
def test_align_loss_args():
    class TestLossFwd(DummyLoss):
        def __init__(self, num_targets, num_agent_inputs, num_head_outputs):
            super(TestLossFwd, self).__init__(num_targets, num_agent_inputs, num_head_outputs)

        def loss_forward(self, F, head_output_2, head_output_1, agent_input_2, target_0, agent_input_1, param1, param2):
            pass

    assert align_loss_args([1, 2, 3], [4, 5, 6, 7], [8, 9], TestLossFwd(3, 4, 2)) == [3, 2, 6, 8, 5]


@pytest.mark.unit_test
def test_to_tuple():
    assert to_tuple(123) == (123,)
    assert to_tuple((1, 2, 3)) == (1, 2, 3)
    assert to_tuple([1, 2, 3]) == (1, 2, 3)


@pytest.mark.unit_test
def test_to_list():
    assert to_list(123) == [123]
    assert to_list((1, 2, 3)) == [1, 2, 3]
    assert to_list([1, 2, 3]) == [1, 2, 3]


@pytest.mark.unit_test
def test_loss_output_dict():
    assert loss_output_dict([1, 2, 3], ['loss', 'loss', 'reg']) == {'loss': [1, 2], 'reg': [3]}


@pytest.mark.unit_test
def test_clip_grad():
    a = np.array([1, 2, -3])
    b = np.array([4, 5, -6])
    clip = 2
    gscale = np.minimum(1.0, clip / np.sqrt(np.sum(np.square(a)) + np.sum(np.square(b))))
    for lhs, rhs in zip(clip_grad([nd.array(a), nd.array(b)], GradientClippingMethod.ClipByGlobalNorm, clip_val=clip),
                        [a, b]):
        assert np.allclose(lhs.asnumpy(), rhs * gscale)
    for lhs, rhs in zip(clip_grad([nd.array(a), nd.array(b)], GradientClippingMethod.ClipByValue, clip_val=clip),
                        [a, b]):
        assert np.allclose(lhs.asnumpy(), np.clip(rhs, -clip, clip))
    for lhs, rhs in zip(clip_grad([nd.array(a), nd.array(b)], GradientClippingMethod.ClipByNorm, clip_val=clip),
                        [a, b]):
        scale = np.minimum(1.0, clip / np.sqrt(np.sum(np.square(rhs))))
        assert np.allclose(lhs.asnumpy(), rhs * scale)


@pytest.mark.unit_test
def test_hybrid_clip():
    x = mx.nd.array((0.5, 1.5, 2.5))
    a = mx.nd.array((1,))
    b = mx.nd.array((2,))
    clipped = hybrid_clip(F=mx.nd, x=x, clip_lower=a, clip_upper=b)
    assert (np.isclose(a=clipped.asnumpy(), b=(1, 1.5, 2))).all()


@pytest.mark.unit_test
def test_broadcast_like():
    x = nd.ones((1, 2)) * 10
    y = nd.ones((100, 100, 2)) * 20
    assert mx.test_utils.almost_equal(x.broadcast_like(y).asnumpy(), broadcast_like(nd, x, y).asnumpy())


@pytest.mark.unit_test
def test_scoped_onxx_enable():
    class Counter(object):
        def __init__(self):
            self._count = 0

        def increment(self):
            self._count += 1

        @property
        def count(self):
            return self._count

    class TempBlock(gluon.HybridBlock, OnnxHandlerBlock):
        def __init__(self, counter: Counter):
            super(TempBlock, self).__init__()
            OnnxHandlerBlock.__init__(self)
            self._counter = counter

        def hybrid_forward(self, F, x, *args, **kwargs):
            if self._onnx:
                self._counter.increment()
            return x

    counter = Counter()
    net = gluon.nn.HybridSequential()
    for _ in range(10):
        net.add(TempBlock(counter))

    # ONNX disabled
    net(nd.zeros((1,)))
    assert counter.count == 0

    # ONNX enabled
    with ScopedOnnxEnable(net):
        net(nd.zeros((1,)))
    assert counter.count == 10
