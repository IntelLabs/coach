import random
import pickle

import pytest
import tensorflow as tf
import numpy as np

from rl_coach.architectures.tensorflow_components.savers import GlobalVariableSaver


def random_name():
    return "%032x" % random.randrange(16 ** 32)


@pytest.fixture
def name():
    return random_name()


@pytest.fixture
def variable(shape, name):
    tf.reset_default_graph()
    return tf.Variable(tf.zeros(shape), name=name)


@pytest.fixture
def shape():
    return (3, 5)


def assert_arrays_ones_shape(arrays, shape, name):
    assert list(arrays.keys()) == [name]
    assert len(arrays) == 1
    assert np.all(list(arrays[name][0]) == np.ones(shape))


@pytest.mark.unit_test
def test_global_variable_saver_to_arrays(variable, name, shape):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(variable.assign(tf.ones(shape)))

        saver = GlobalVariableSaver("name")
        arrays = saver.to_arrays(session)
        assert_arrays_ones_shape(arrays, shape, name)


@pytest.mark.unit_test
def test_global_variable_saver_from_arrays(variable, name, shape):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver = GlobalVariableSaver("name")
        saver.from_arrays(session, {name: np.ones(shape)})
        arrays = saver.to_arrays(session)
        assert_arrays_ones_shape(arrays, shape, name)


@pytest.mark.unit_test
def test_global_variable_saver_to_string(variable, name, shape):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(variable.assign(tf.ones(shape)))

        saver = GlobalVariableSaver("name")
        string = saver.to_string(session)
        arrays = pickle.loads(string)
        assert_arrays_ones_shape(arrays, shape, name)


@pytest.mark.unit_test
def test_global_variable_saver_from_string(variable, name, shape):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver = GlobalVariableSaver("name")
        saver.from_string(session, pickle.dumps({name: np.ones(shape)}, protocol=-1))
        arrays = saver.to_arrays(session)
        assert_arrays_ones_shape(arrays, shape, name)
