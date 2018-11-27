import mxnet as mx
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.architectures.mxnet_components.middlewares.lstm_middleware import LSTMMiddleware


@pytest.mark.unit_test
def test_lstm_middleware():
    params = LSTMMiddlewareParameters(number_of_lstm_cells=25, scheme=MiddlewareScheme.Medium)
    mid = LSTMMiddleware(params=params)
    mid.initialize()
    # NTC
    embedded_data = mx.nd.random.uniform(low=0, high=1, shape=(10, 15, 20))
    # NTC -> TNC
    output = mid(embedded_data)
    assert output.ndim == 3  # since last block was flatten
    assert output.shape[0] == 15  # since t is 15
    assert output.shape[1] == 10  # since batch_size is 10
    assert output.shape[2] == 25  # since number_of_lstm_cells is 25
