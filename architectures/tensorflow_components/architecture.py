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

from architectures.architecture import Architecture
import tensorflow as tf
from utils import force_list, squeeze_list
from configurations import Preset, MiddlewareTypes
import numpy as np
import time


class TensorFlowArchitecture(Architecture):
    def __init__(self, tuning_parameters, name="", global_network=None, network_is_local=True):
        """
        :param tuning_parameters: The parameters used for running the algorithm
        :type tuning_parameters: Preset
        :param name: The name of the network
        """
        Architecture.__init__(self, tuning_parameters, name)
        self.middleware_embedder = None
        self.network_is_local = network_is_local
        assert tuning_parameters.agent.tensorflow_support, 'TensorFlow is not supported for this agent'
        self.sess = tuning_parameters.sess
        self.inputs = []
        self.outputs = []
        self.targets = []
        self.losses = []
        self.total_loss = None
        self.trainable_weights = []
        self.weights_placeholders = []
        self.curr_rnn_c_in = None
        self.curr_rnn_h_in = None
        self.gradients_wrt_inputs = []

        self.optimizer_type = self.tp.agent.optimizer_type
        if self.tp.seed is not None:
            tf.set_random_seed(self.tp.seed)
        with tf.variable_scope(self.name, initializer=tf.contrib.layers.xavier_initializer()):
            self.global_step = tf.train.get_or_create_global_step()

            # build the network
            self.get_model(tuning_parameters)

            # model weights
            self.trainable_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            # locks for synchronous training
            if self.tp.distributed and not self.tp.agent.async_training and not self.network_is_local:
                self.lock_counter = tf.get_variable("lock_counter", [], tf.int32,
                                                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                    trainable=False)
                self.lock = self.lock_counter.assign_add(1, use_locking=True)
                self.lock_init = self.lock_counter.assign(0)

                self.release_counter = tf.get_variable("release_counter", [], tf.int32,
                                                       initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                       trainable=False)
                self.release = self.release_counter.assign_add(1, use_locking=True)
                self.release_init = self.release_counter.assign(0)

            # local network does the optimization so we need to create all the ops we are going to use to optimize
            for idx, var in enumerate(self.trainable_weights):
                placeholder = tf.placeholder(tf.float32, shape=var.get_shape(), name=str(idx) + '_holder')
                self.weights_placeholders.append(placeholder)
            self.update_weights_from_list = [weights.assign(holder) for holder, weights in
                                             zip(self.weights_placeholders, self.trainable_weights)]

            # gradients ops
            self.tensor_gradients = tf.gradients(self.total_loss, self.trainable_weights)
            self.gradients_norm = tf.global_norm(self.tensor_gradients)
            if self.tp.clip_gradients is not None and self.tp.clip_gradients != 0:
                self.clipped_grads, self.grad_norms = tf.clip_by_global_norm(self.tensor_gradients,
                                                                             tuning_parameters.clip_gradients)

            # gradients of the outputs w.r.t. the inputs
            # at the moment, this is only used by ddpg
            if len(self.outputs) == 1:
                self.gradients_wrt_inputs = [tf.gradients(self.outputs[0], input_ph) for input_ph in self.inputs]
                self.gradients_weights_ph = tf.placeholder('float32', self.outputs[0].shape, 'output_gradient_weights')
                self.weighted_gradients = tf.gradients(self.outputs[0], self.trainable_weights, self.gradients_weights_ph)

            # L2 regularization
            if self.tp.agent.l2_regularization != 0:
                self.l2_regularization = [tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_weights])
                                          * self.tp.agent.l2_regularization]
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l2_regularization)

            self.inc_step = self.global_step.assign_add(1)

            # defining the optimization process (for LBFGS we have less control over the optimizer)
            if self.optimizer_type != 'LBFGS':
                # no global network, this is a plain simple centralized training
                self.update_weights_from_batch_gradients = self.optimizer.apply_gradients(
                    zip(self.weights_placeholders, self.trainable_weights), global_step=self.global_step)

            # initialize or restore model
            if not self.tp.distributed:
                self.init_op = tf.global_variables_initializer()

                if self.sess:
                        self.sess.run(self.init_op)

        self.accumulated_gradients = None

    def reset_accumulated_gradients(self):
        """
        Reset the gradients accumulation placeholder
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = self.tp.sess.run(self.trainable_weights)

        for ix, grad in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[ix] = grad * 0

    def accumulate_gradients(self, inputs, targets, additional_fetches=None):
        """
        Runs a forward pass & backward pass, clips gradients if needed and accumulates them into the accumulation
        placeholders
        :param additional_fetches: Optional tensors to fetch during gradients calculation
        :param inputs: The input batch for the network
        :param targets: The targets corresponding to the input batch
        :return: A list containing the total loss and the individual network heads losses
        """

        if self.accumulated_gradients is None:
            self.reset_accumulated_gradients()

        # feed inputs
        if additional_fetches is None:
            additional_fetches = []
        inputs = force_list(inputs)

        feed_dict = dict(zip(self.inputs, inputs))

        # feed targets
        targets = force_list(targets)
        for placeholder_idx, target in enumerate(targets):
            feed_dict[self.targets[placeholder_idx]] = target

        if self.optimizer_type != 'LBFGS':
            # set the fetches
            fetches = [self.gradients_norm]
            if self.tp.clip_gradients:
                fetches.append(self.clipped_grads)
            else:
                fetches.append(self.tensor_gradients)
            fetches += [self.total_loss, self.losses]
            if self.tp.agent.middleware_type == MiddlewareTypes.LSTM:
                fetches.append(self.middleware_embedder.state_out)
            additional_fetches_start_idx = len(fetches)
            fetches += additional_fetches

            # feed the lstm state if necessary
            if self.tp.agent.middleware_type == MiddlewareTypes.LSTM:
                # we can't always assume that we are starting from scratch here can we?
                feed_dict[self.middleware_embedder.c_in] = self.middleware_embedder.c_init
                feed_dict[self.middleware_embedder.h_in] = self.middleware_embedder.h_init

            # get grads
            result = self.tp.sess.run(fetches, feed_dict=feed_dict)

            # extract the fetches
            norm_unclipped_grads, grads, total_loss, losses = result[:4]
            if self.tp.agent.middleware_type == MiddlewareTypes.LSTM:
                (self.curr_rnn_c_in, self.curr_rnn_h_in) = result[4]
            fetched_tensors = []
            if len(additional_fetches) > 0:
                fetched_tensors = result[additional_fetches_start_idx:]

            # accumulate the gradients
            for idx, grad in enumerate(grads):
                self.accumulated_gradients[idx] += grad

            return total_loss, losses, norm_unclipped_grads, fetched_tensors

        else:
            self.optimizer.minimize(session=self.tp.sess, feed_dict=feed_dict)

            return [0]

    def apply_and_reset_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights and resets the accumulation placeholder
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        self.apply_gradients(gradients, scaler)
        self.reset_accumulated_gradients()

    def apply_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        if self.tp.agent.async_training or not self.tp.distributed:
            if hasattr(self, 'global_step') and not self.network_is_local:
                self.tp.sess.run(self.inc_step)

        if self.optimizer_type != 'LBFGS':

            # lock barrier
            if hasattr(self, 'lock_counter'):
                self.tp.sess.run(self.lock)
                while self.tp.sess.run(self.lock_counter) % self.tp.num_threads != 0:
                    time.sleep(0.00001)
                # rescale the gradients so that they average out with the gradients from the other workers
                scaler /= float(self.tp.num_threads)

            # apply gradients
            if scaler != 1.:
                for gradient in gradients:
                    gradient /= scaler
            feed_dict = dict(zip(self.weights_placeholders, gradients))
            _ = self.tp.sess.run(self.update_weights_from_batch_gradients, feed_dict=feed_dict)

            # release barrier
            if hasattr(self, 'release_counter'):
                self.tp.sess.run(self.release)
                while self.tp.sess.run(self.release_counter) % self.tp.num_threads != 0:
                    time.sleep(0.00001)

    def predict(self, inputs, outputs=None):
        """
        Run a forward pass of the network using the given input
        :param inputs: The input for the network
        :param outputs: The output for the network, defaults to self.outputs
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """

        feed_dict = dict(zip(self.inputs, force_list(inputs)))
        if outputs is None:
            outputs = self.outputs

        if self.tp.agent.middleware_type == MiddlewareTypes.LSTM:
            feed_dict[self.middleware_embedder.c_in] = self.curr_rnn_c_in
            feed_dict[self.middleware_embedder.h_in] = self.curr_rnn_h_in

            output, (self.curr_rnn_c_in, self.curr_rnn_h_in) = self.tp.sess.run([outputs, self.middleware_embedder.state_out], feed_dict=feed_dict)
        else:
            output = self.tp.sess.run(outputs, feed_dict)

        return squeeze_list(output)

    def train_on_batch(self, inputs, targets, scaler=1., additional_fetches=None):
        """
        Given a batch of examples and targets, runs a forward pass & backward pass and then applies the gradients
        :param additional_fetches: Optional tensors to fetch during the training process
        :param inputs: The input for the network
        :param targets: The targets corresponding to the input batch
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        :return: The loss of the network
        """
        if additional_fetches is None:
            additional_fetches = []
        force_list(additional_fetches)
        loss = self.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches)
        self.apply_and_reset_gradients(self.accumulated_gradients, scaler)
        return loss

    def get_weights(self):
        """
        :return: a list of tensors containing the network weights for each layer
        """
        return self.trainable_weights

    def set_weights(self, weights, new_rate=1.0):
        """
        Sets the network weights from the given list of weights tensors
        """
        feed_dict = {}
        old_weights, new_weights = self.tp.sess.run([self.get_weights(), weights])
        for placeholder_idx, new_weight in enumerate(new_weights):
            feed_dict[self.weights_placeholders[placeholder_idx]]\
                = new_rate * new_weight + (1 - new_rate) * old_weights[placeholder_idx]
        self.tp.sess.run(self.update_weights_from_list, feed_dict)

    def write_graph_to_logdir(self, summary_dir):
        """
        Writes the tensorflow graph to the logdir for tensorboard visualization
        :param summary_dir: the path to the logdir
        """
        summary_writer = tf.summary.FileWriter(summary_dir)
        summary_writer.add_graph(self.sess.graph)

    def get_variable_value(self, variable):
        """
        Get the value of a variable from the graph
        :param variable: the variable
        :return: the value of the variable
        """
        return self.sess.run(variable)

    def set_variable_value(self, assign_op, value, placeholder=None):
        """
        Updates the value of a variable.
        This requires having an assign operation for the variable, and a placeholder which will provide the value
        :param assign_op: an assign operation for the variable
        :param value: a value to set the variable to
        :param placeholder: a placeholder to hold the given value for injecting it into the variable
        """
        self.sess.run(assign_op, feed_dict={placeholder: value})
