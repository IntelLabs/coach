import mxnet as mx
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.mxnet_components.middlewares.fc_middleware import FCMiddleware


@pytest.mark.unit_test
def test_fc_middleware():
    params = FCMiddlewareParameters(scheme=MiddlewareScheme.Medium)
    mid = FCMiddleware(params=params)
    mid.initialize()
    embedded_data = mx.nd.random.uniform(low=0, high=1, shape=(10, 100))
    output = mid(embedded_data)
    assert output.ndim == 2  # since last block was flatten
    assert output.shape[0] == 10  # since batch_size is 10
    assert output.shape[1] == 512  # since last layer of middleware (middle scheme) had 512 units
