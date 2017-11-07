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

import tensorflow as tf
import numpy as np


class SharedRunningStats(object):
    def __init__(self, tuning_parameters, replicated_device, epsilon=1e-2, shape=(), name=""):
        self.tp = tuning_parameters
        with tf.device(replicated_device):
            with tf.variable_scope(name):
                self._sum = tf.get_variable(
                    dtype=tf.float64,
                    shape=shape,
                    initializer=tf.constant_initializer(0.0),
                    name="running_sum", trainable=False)
                self._sum_squared = tf.get_variable(
                    dtype=tf.float64,
                    shape=shape,
                    initializer=tf.constant_initializer(epsilon),
                    name="running_sum_squared", trainable=False)
                self._count = tf.get_variable(
                    dtype=tf.float64,
                    shape=(),
                    initializer=tf.constant_initializer(epsilon),
                    name="count", trainable=False)

                self._shape = shape
                self._mean = self._sum / self._count
                self._std = tf.sqrt(tf.maximum((self._sum_squared - self._count*tf.square(self._mean))
                                               / tf.maximum(self._count-1, 1), epsilon))

                self.new_sum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
                self.new_sum_squared = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
                self.newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')

                self._inc_sum = tf.assign_add(self._sum, self.new_sum, use_locking=True)
                self._inc_sum_squared = tf.assign_add(self._sum_squared, self.new_sum_squared, use_locking=True)
                self._inc_count = tf.assign_add(self._count, self.newcount, use_locking=True)

    def push(self, x):
        x = x.astype('float64')
        self.tp.sess.run([self._inc_sum, self._inc_sum_squared, self._inc_count],
                         feed_dict={
                             self.new_sum: x.sum(axis=0).ravel(),
                             self.new_sum_squared: np.square(x).sum(axis=0).ravel(),
                             self.newcount: np.array(len(x), dtype='float64')
                         })

    @property
    def n(self):
        return self.tp.sess.run(self._count)

    @property
    def mean(self):
        return self.tp.sess.run(self._mean)

    @property
    def var(self):
        return self.std ** 2

    @property
    def std(self):
        return self.tp.sess.run(self._std)

    @property
    def shape(self):
        return self._shape