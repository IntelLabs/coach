import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np
from rl_coach.architectures.tensorflow_components.embedders.vector_embedder import VectorEmbedder, EmbedderScheme
import tensorflow as tf
from tensorflow import logging

logging.set_verbosity(logging.INFO)

@pytest.fixture
def reset():
    tf.reset_default_graph()


@pytest.mark.unit_test
def test_embedder(reset):
    # creating a vector embedder with a matrix
    with pytest.raises(ValueError):
        embedder = VectorEmbedder(np.array([10, 10]), name="test")

    # creating a simple vector embedder
    embedder = VectorEmbedder(np.array([10]), name="test")

    # make sure the ops where not created yet
    assert len(tf.get_default_graph().get_operations()) == 0

    # call the embedder
    input_ph, output_ph = embedder()

    # make sure that now the ops were created
    assert len(tf.get_default_graph().get_operations()) > 0

    # try feeding a batch of one example
    input = np.random.rand(1, 10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(embedder.output, {embedder.input: input})
    assert output.shape == (1, 256)

    # now make sure the returned placeholders behave the same
    output = sess.run(output_ph, {input_ph: input})
    assert output.shape == (1, 256)

    # make sure the naming is correct
    assert embedder.get_name() == "test"


@pytest.mark.unit_test
def test_complex_embedder(reset):
    # creating a deep vector embedder
    embedder = VectorEmbedder(np.array([10]), name="test", scheme=EmbedderScheme.Deep)

    # call the embedder
    embedder()

    # try feeding a batch of one example
    input = np.random.rand(1, 10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(embedder.output, {embedder.input: input})
    assert output.shape == (1, 128)  # should have flattened the input


@pytest.mark.unit_test
def test_activation_function(reset):
    # creating a deep vector embedder with relu
    embedder = VectorEmbedder(np.array([10]), name="relu", scheme=EmbedderScheme.Deep,
                              activation_function=tf.nn.relu)

    # call the embedder
    embedder()

    # try feeding a batch of one example
    input = np.random.rand(1, 10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(embedder.output, {embedder.input: input})
    assert np.all(output >= 0)  # should have flattened the input

    # creating a deep vector embedder with tanh
    embedder_tanh = VectorEmbedder(np.array([10]), name="tanh", scheme=EmbedderScheme.Deep,
                                   activation_function=tf.nn.tanh)

    # call the embedder
    embedder_tanh()

    # try feeding a batch of one example
    input = np.random.rand(1, 10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(embedder_tanh.output, {embedder_tanh.input: input})
    assert np.all(output >= -1) and np.all(output <= 1)
