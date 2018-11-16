import mxnet as mx
import numpy as np
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.architectures.mxnet_components.heads.head import NormalizedRSSInitializer


@pytest.mark.unit_test
def test_normalized_rss_initializer():
    target_rss = 0.5
    units = 10
    dense = mx.gluon.nn.Dense(units=units, weight_initializer=NormalizedRSSInitializer(target_rss))
    dense.initialize()

    input_data = mx.random.uniform(shape=(25, 5))
    output_data = dense(input_data)

    weights = dense.weight.data()
    assert weights.shape == (10, 5)
    rss = weights.square().sum(axis=1).sqrt()
    np.testing.assert_almost_equal(rss.asnumpy(), np.tile(target_rss, units))
