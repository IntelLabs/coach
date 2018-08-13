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

import numpy as np
import tensorflow as tf


class SharedRunningStats(object):
    def __init__(self, replicated_device=None, epsilon=1e-2, name="", create_ops=True):
        self.sess = None
        self.name = name
        self.replicated_device = replicated_device
        self.epsilon = epsilon
        self.ops_were_created = False
        if create_ops:
            with tf.device(replicated_device):
                self.create_ops()

    def create_ops(self, shape=[1], clip_values=None):
        self.clip_values = clip_values
        with tf.variable_scope(self.name):
            self._sum = tf.get_variable(
                dtype=tf.float64,
                initializer=tf.constant_initializer(0.0),
                name="running_sum", trainable=False, shape=shape, validate_shape=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            self._sum_squared = tf.get_variable(
                dtype=tf.float64,
                initializer=tf.constant_initializer(self.epsilon),
                name="running_sum_squared", trainable=False, shape=shape, validate_shape=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            self._count = tf.get_variable(
                dtype=tf.float64,
                shape=(),
                initializer=tf.constant_initializer(self.epsilon),
                name="count", trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])

            self._shape = None
            self._mean = tf.div(self._sum, self._count, name="mean")
            self._std = tf.sqrt(tf.maximum((self._sum_squared - self._count*tf.square(self._mean))
                                           / tf.maximum(self._count-1, 1), self.epsilon), name="stdev")
            self.tf_mean = tf.cast(self._mean, 'float32')
            self.tf_std = tf.cast(self._std, 'float32')

            self.new_sum = tf.placeholder(dtype=tf.float64, name='sum')
            self.new_sum_squared = tf.placeholder(dtype=tf.float64, name='var')
            self.newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')

            self._inc_sum = tf.assign_add(self._sum, self.new_sum, use_locking=True)
            self._inc_sum_squared = tf.assign_add(self._sum_squared, self.new_sum_squared, use_locking=True)
            self._inc_count = tf.assign_add(self._count, self.newcount, use_locking=True)

            self.raw_obs = tf.placeholder(dtype=tf.float64, name='raw_obs')
            self.normalized_obs = (self.raw_obs - self._mean) / self._std
            if self.clip_values is not None:
                self.clipped_obs = tf.clip_by_value(self.normalized_obs, self.clip_values[0], self.clip_values[1])

            self.ops_were_created = True

    def set_session(self, sess):
        self.sess = sess

    def push(self, x):
        x = x.astype('float64')
        self.sess.run([self._inc_sum, self._inc_sum_squared, self._inc_count],
                         feed_dict={
                             self.new_sum: x.sum(axis=0).ravel(),
                             self.new_sum_squared: np.square(x).sum(axis=0).ravel(),
                             self.newcount: np.array(len(x), dtype='float64')
                         })
        if self._shape is None:
            self._shape = x.shape

    @property
    def n(self):
        return self.sess.run(self._count)

    @property
    def mean(self):
        return self.sess.run(self._mean)

    @property
    def var(self):
        return self.std ** 2

    @property
    def std(self):
        return self.sess.run(self._std)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = val
        self.new_sum.set_shape(val)
        self.new_sum_squared.set_shape(val)
        self.tf_mean.set_shape(val)
        self.tf_std.set_shape(val)
        self._sum.set_shape(val)
        self._sum_squared.set_shape(val)

    def normalize(self, batch):
        if self.clip_values is not None:
            return self.sess.run(self.clipped_obs, feed_dict={self.raw_obs: batch})
        else:
            return self.sess.run(self.normalized_obs, feed_dict={self.raw_obs: batch})
