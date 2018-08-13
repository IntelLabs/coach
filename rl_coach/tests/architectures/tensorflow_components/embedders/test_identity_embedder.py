import os
import sys

from rl_coach.base_parameters import EmbedderScheme

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np
from rl_coach.architectures.tensorflow_components.embedders.vector_embedder import VectorEmbedder
import tensorflow as tf
from tensorflow import logging

logging.set_verbosity(logging.INFO)

@pytest.fixture
def reset():
    tf.reset_default_graph()


@pytest.mark.unit_test
def test_embedder(reset):
    embedder = VectorEmbedder(np.array([10, 10]), name="test", scheme=EmbedderScheme.Empty)

    # make sure the ops where not created yet
    assert len(tf.get_default_graph().get_operations()) == 0

    # call the embedder
    input_ph, output_ph = embedder()

    # make sure that now the ops were created
    assert len(tf.get_default_graph().get_operations()) > 0

    # try feeding a batch of one example  # TODO: consider auto converting to batch
    input = np.random.rand(1, 10, 10)
    sess = tf.Session()
    output = sess.run(embedder.output, {embedder.input: input})
    assert output.shape == (1, 100)  # should have flattened the input

    # now make sure the returned placeholders behave the same
    output = sess.run(output_ph, {input_ph: input})
    assert output.shape == (1, 100)  # should have flattened the input

    # make sure the naming is correct
    assert embedder.get_name() == "test"
