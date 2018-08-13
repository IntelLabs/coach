# nasty hack to deal with issue #46
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np

import time
from rl_coach.memories.non_episodic.differentiable_neural_dictionary import QDND
import tensorflow as tf

NUM_ACTIONS = 3
NUM_DND_ENTRIES_TO_ADD = 10000
EMBEDDING_SIZE = 512
NUM_SAMPLED_EMBEDDINGS = 500
NUM_NEIGHBORS = 10
DND_SIZE = 500000

@pytest.fixture()
def dnd():
    return QDND(
                DND_SIZE,
                EMBEDDING_SIZE,
                NUM_ACTIONS,
                0.1,
                key_error_threshold=0,
                learning_rate=0.0001,
                num_neighbors=NUM_NEIGHBORS
                )


@pytest.mark.unit_test
def test_random_sample_from_dnd(dnd: QDND):
    # store single non terminal transition
    embeddings = [np.random.rand(EMBEDDING_SIZE) for j in range(NUM_DND_ENTRIES_TO_ADD)]
    actions = [np.random.randint(NUM_ACTIONS) for j in range(NUM_DND_ENTRIES_TO_ADD)]
    values = [np.random.rand() for j in range(NUM_DND_ENTRIES_TO_ADD)]
    dnd.add(embeddings, actions, values)
    dnd_embeddings, dnd_values, dnd_indices = dnd.query(embeddings[0:10], 0, NUM_NEIGHBORS)

    # calculate_normalization_factor
    sampled_embeddings = dnd.sample_embeddings(NUM_SAMPLED_EMBEDDINGS)
    coefficient = 1/(NUM_SAMPLED_EMBEDDINGS * (NUM_SAMPLED_EMBEDDINGS - 1.0))
    tf_current_embedding = tf.placeholder(tf.float32, shape=(EMBEDDING_SIZE), name='current_embedding')
    tf_other_embeddings = tf.placeholder(tf.float32, shape=(NUM_SAMPLED_EMBEDDINGS - 1, EMBEDDING_SIZE), name='other_embeddings')

    sub = tf_current_embedding - tf_other_embeddings
    square = tf.square(sub)
    result = tf.reduce_sum(square)



    ###########################
    # more efficient method
    ###########################
    sampled_embeddings_expanded = tf.placeholder(
        tf.float32, shape=(1, NUM_SAMPLED_EMBEDDINGS, EMBEDDING_SIZE), name='sampled_embeddings_expanded')
    sampled_embeddings_tiled = tf.tile(sampled_embeddings_expanded, (sampled_embeddings_expanded.shape[1], 1, 1))
    sampled_embeddings_transposed = tf.transpose(sampled_embeddings_tiled, (1, 0, 2))
    sub2 = sampled_embeddings_tiled - sampled_embeddings_transposed
    square2 = tf.square(sub2)
    result2 = tf.reduce_sum(square2)

    config = tf.ConfigProto()
    config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
    config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed

    sess = tf.Session(config=config)

    sum1 = 0
    start = time.time()
    for i in range(NUM_SAMPLED_EMBEDDINGS):
        curr_sampled_embedding = sampled_embeddings[i]
        other_embeddings = np.delete(sampled_embeddings, i, 0)
        sum1 += sess.run(result, feed_dict={tf_current_embedding: curr_sampled_embedding, tf_other_embeddings: other_embeddings})
    print("1st method: {} sec".format(time.time()-start))

    start = time.time()
    sum2 = sess.run(result2, feed_dict={sampled_embeddings_expanded: np.expand_dims(sampled_embeddings,0)})
    print("2nd method: {} sec".format(time.time()-start))

    # validate that results are equal
    print("sum1 = {}, sum2 = {}".format(sum1, sum2))

    norm_factor = -0.5/(coefficient * sum2)

if __name__ == '__main__':
    test_random_sample_from_dnd(dnd())

