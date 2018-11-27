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
import os
import time
from typing import Any, List, Tuple, Dict

import numpy as np
import tensorflow as tf

from rl_coach.architectures.architecture import Architecture
from rl_coach.architectures.tensorflow_components.savers import GlobalVariableSaver
from rl_coach.base_parameters import AgentParameters, DistributedTaskParameters
from rl_coach.core_types import GradientClippingMethod
from rl_coach.saver import SaverCollection
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import force_list, squeeze_list, start_shell_command_and_wait


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        layer_weight_name = '_'.join(var.name.split('/')[-3:])[:-2]

        with tf.name_scope(layer_weight_name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def local_getter(getter, name, *args, **kwargs):
    """
    This is a wrapper around the tf.get_variable function which puts the variables in the local variables collection
    instead of the global variables collection. The local variables collection will hold variables which are not shared
    between workers. these variables are also assumed to be non-trainable (the optimizer does not apply gradients to
    these variables), but we can calculate the gradients wrt these variables, and we can update their content.
    """
    kwargs['collections'] = [tf.GraphKeys.LOCAL_VARIABLES]
    return getter(name, *args, **kwargs)


class TensorFlowArchitecture(Architecture):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, name: str= "",
                 global_network=None, network_is_local: bool=True, network_is_trainable: bool=False):
        """
        :param agent_parameters: the agent parameters
        :param spaces: the spaces definition of the agent
        :param name: the name of the network
        :param global_network: the global network replica that is shared between all the workers
        :param network_is_local: is the network global (shared between workers) or local (dedicated to the worker)
        :param network_is_trainable: is the network trainable (we can apply gradients on it)
        """
        super().__init__(agent_parameters, spaces, name)
        self.middleware = None
        self.network_is_local = network_is_local
        self.global_network = global_network
        if not self.network_parameters.tensorflow_support:
            raise ValueError('TensorFlow is not supported for this agent')
        self.sess = None
        self.inputs = {}
        self.outputs = []
        self.targets = []
        self.importance_weights = []
        self.losses = []
        self.total_loss = None
        self.trainable_weights = []
        self.weights_placeholders = []
        self.shared_accumulated_gradients = []
        self.curr_rnn_c_in = None
        self.curr_rnn_h_in = None
        self.gradients_wrt_inputs = []
        self.train_writer = None
        self.accumulated_gradients = None
        self.network_is_trainable = network_is_trainable

        self.is_chief = self.ap.task_parameters.task_index == 0
        self.network_is_global = not self.network_is_local and global_network is None
        self.distributed_training = self.network_is_global or self.network_is_local and global_network is not None

        self.optimizer_type = self.network_parameters.optimizer_type
        if self.ap.task_parameters.seed is not None:
            tf.set_random_seed(self.ap.task_parameters.seed)
        with tf.variable_scope("/".join(self.name.split("/")[1:]), initializer=tf.contrib.layers.xavier_initializer(),
                               custom_getter=local_getter if network_is_local and global_network else None):
            self.global_step = tf.train.get_or_create_global_step()

            # build the network
            self.get_model()

            # model weights
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.full_name)

            # create the placeholder for the assigning gradients and some tensorboard summaries for the weights
            for idx, var in enumerate(self.weights):
                placeholder = tf.placeholder(tf.float32, shape=var.get_shape(), name=str(idx) + '_holder')
                self.weights_placeholders.append(placeholder)
                if self.ap.visualization.tensorboard:
                    variable_summaries(var)

            # create op for assigning a list of weights to the network weights
            self.update_weights_from_list = [weights.assign(holder) for holder, weights in
                                             zip(self.weights_placeholders, self.weights)]

            # locks for synchronous training
            if self.network_is_global:
                self._create_locks_for_synchronous_training()

            # gradients ops
            self._create_gradient_ops()

            # L2 regularization
            if self.network_parameters.l2_regularization != 0:
                self.l2_regularization = [tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
                                          * self.network_parameters.l2_regularization]
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.l2_regularization)

            self.inc_step = self.global_step.assign_add(1)

            # reset LSTM hidden cells
            self.reset_internal_memory()

            if self.ap.visualization.tensorboard:
                current_scope_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                            scope=tf.contrib.framework.get_name_scope())
                self.merged = tf.summary.merge(current_scope_summaries)

            # initialize or restore model
            self.init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            # set the fetches for training
            self._set_initial_fetch_list()

    def get_model(self) -> None:
        """
        Constructs the model using `network_parameters` and sets `input_embedders`, `middleware`,
        `output_heads`, `outputs`, `losses`, `total_loss`, `adaptive_learning_rate_scheme`,
        `current_learning_rate`, and `optimizer`
        """
        raise NotImplementedError

    def _set_initial_fetch_list(self):
        """
        Create an initial list of tensors to fetch in each training iteration
        :return: None
        """
        self.train_fetches = [self.gradients_norm]
        if self.network_parameters.clip_gradients:
            self.train_fetches.append(self.clipped_grads)
        else:
            self.train_fetches.append(self.tensor_gradients)
        self.train_fetches += [self.total_loss, self.losses]
        if self.middleware.__class__.__name__ == 'LSTMMiddleware':
            self.train_fetches.append(self.middleware.state_out)
        self.additional_fetches_start_idx = len(self.train_fetches)

    def _create_locks_for_synchronous_training(self):
        """
        Create locks for synchronizing the different workers during training
        :return: None
        """
        self.lock_counter = tf.get_variable("lock_counter", [], tf.int32,
                                            initializer=tf.constant_initializer(0, dtype=tf.int32),
                                            trainable=False)
        self.lock = self.lock_counter.assign_add(1, use_locking=True)
        self.lock_init = self.lock_counter.assign(0)

        self.release_counter = tf.get_variable("release_counter", [], tf.int32,
                                               initializer=tf.constant_initializer(0, dtype=tf.int32),
                                               trainable=False)
        self.release = self.release_counter.assign_add(1, use_locking=True)
        self.release_decrement = self.release_counter.assign_add(-1, use_locking=True)
        self.release_init = self.release_counter.assign(0)

    def _create_gradient_ops(self):
        """
        Create all the tensorflow operations for calculating gradients, processing the gradients and applying them
        :return: None
        """

        self.tensor_gradients = tf.gradients(self.total_loss, self.weights)
        self.gradients_norm = tf.global_norm(self.tensor_gradients)

        # gradient clipping
        if self.network_parameters.clip_gradients is not None and self.network_parameters.clip_gradients != 0:
            self._create_gradient_clipping_ops()

        # when using a shared optimizer, we create accumulators to store gradients from all the workers before
        # applying them
        if self.distributed_training:
            self._create_gradient_accumulators()

        # gradients of the outputs w.r.t. the inputs
        # at the moment, this is only used by ddpg
        self.gradients_wrt_inputs = [{name: tf.gradients(output, input_ph) for name, input_ph in
                                      self.inputs.items()} for output in self.outputs]
        self.gradients_weights_ph = [tf.placeholder('float32', self.outputs[i].shape, 'output_gradient_weights')
                                     for i in range(len(self.outputs))]
        self.weighted_gradients = []
        for i in range(len(self.outputs)):
            unnormalized_gradients = tf.gradients(self.outputs[i], self.weights, self.gradients_weights_ph[i])
            # unnormalized gradients seems to be better at the time. TODO: validate this accross more environments
            # self.weighted_gradients.append(list(map(lambda x: tf.div(x, self.network_parameters.batch_size),
            #                                         unnormalized_gradients)))
            self.weighted_gradients.append(unnormalized_gradients)

        # defining the optimization process (for LBFGS we have less control over the optimizer)
        if self.optimizer_type != 'LBFGS' and self.network_is_trainable:
            self._create_gradient_applying_ops()

    def _create_gradient_accumulators(self):
        if self.network_is_global:
            self.shared_accumulated_gradients = [tf.Variable(initial_value=tf.zeros_like(var)) for var in self.weights]
            self.accumulate_shared_gradients = [var.assign_add(holder, use_locking=True) for holder, var in
                                                zip(self.weights_placeholders, self.shared_accumulated_gradients)]
            self.init_shared_accumulated_gradients = [var.assign(tf.zeros_like(var)) for var in
                                                      self.shared_accumulated_gradients]
        elif self.network_is_local:
            self.accumulate_shared_gradients = self.global_network.accumulate_shared_gradients
            self.init_shared_accumulated_gradients = self.global_network.init_shared_accumulated_gradients

    def _create_gradient_clipping_ops(self):
        """
        Create tensorflow ops for clipping the gradients according to the given GradientClippingMethod
        :return: None
        """
        if self.network_parameters.gradients_clipping_method == GradientClippingMethod.ClipByGlobalNorm:
            self.clipped_grads, self.grad_norms = tf.clip_by_global_norm(self.tensor_gradients,
                                                                         self.network_parameters.clip_gradients)
        elif self.network_parameters.gradients_clipping_method == GradientClippingMethod.ClipByValue:
            self.clipped_grads = [tf.clip_by_value(grad,
                                                   -self.network_parameters.clip_gradients,
                                                   self.network_parameters.clip_gradients)
                                  for grad in self.tensor_gradients]
        elif self.network_parameters.gradients_clipping_method == GradientClippingMethod.ClipByNorm:
            self.clipped_grads = [tf.clip_by_norm(grad, self.network_parameters.clip_gradients)
                                  for grad in self.tensor_gradients]

    def _create_gradient_applying_ops(self):
        """
        Create tensorflow ops for applying the gradients to the network weights according to the training scheme
        (distributed training - local or global network, shared optimizer, etc.)
        :return: None
        """
        if self.network_is_global and self.network_parameters.shared_optimizer and \
                not self.network_parameters.async_training:
            # synchronous training with shared optimizer? -> create an operation for applying the gradients
            # accumulated in the shared gradients accumulator
            self.update_weights_from_shared_gradients = self.optimizer.apply_gradients(
                zip(self.shared_accumulated_gradients, self.weights),
                global_step=self.global_step)

        elif self.distributed_training and self.network_is_local:
            # distributed training but independent optimizer? -> create an operation for applying the gradients
            # to the global weights
            self.update_weights_from_batch_gradients = self.optimizer.apply_gradients(
                zip(self.weights_placeholders, self.global_network.weights), global_step=self.global_step)

        elif self.network_is_trainable:
            # not any of the above but is trainable? -> create an operation for applying the gradients to
            # this network weights
            self.update_weights_from_batch_gradients = self.optimizer.apply_gradients(
                zip(self.weights_placeholders, self.weights), global_step=self.global_step)

    def set_session(self, sess):
        self.sess = sess

        task_is_distributed = isinstance(self.ap.task_parameters, DistributedTaskParameters)
        # initialize the session parameters in single threaded runs. Otherwise, this is done through the
        # MonitoredSession object in the graph manager
        if not task_is_distributed:
            self.sess.run(self.init_op)

        if self.ap.visualization.tensorboard:
            # Write the merged summaries to the current experiment directory
            if not task_is_distributed:
                self.train_writer = tf.summary.FileWriter(self.ap.task_parameters.experiment_path + '/tensorboard')
                self.train_writer.add_graph(self.sess.graph)
            elif self.network_is_local:
                self.train_writer = tf.summary.FileWriter(self.ap.task_parameters.experiment_path +
                                                          '/tensorboard/worker{}'.format(self.ap.task_parameters.task_index))
                self.train_writer.add_graph(self.sess.graph)

        # wait for all the workers to set their session
        if not self.network_is_local:
            self.wait_for_all_workers_barrier()

    def reset_accumulated_gradients(self):
        """
        Reset the gradients accumulation placeholder
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = self.sess.run(self.weights)

        for ix, grad in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[ix] = grad * 0

    def accumulate_gradients(self, inputs, targets, additional_fetches=None, importance_weights=None,
                             no_accumulation=False):
        """
        Runs a forward pass & backward pass, clips gradients if needed and accumulates them into the accumulation
        placeholders
        :param additional_fetches: Optional tensors to fetch during gradients calculation
        :param inputs: The input batch for the network
        :param targets: The targets corresponding to the input batch
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :param no_accumulation: If is set to True, the gradients in the accumulated gradients placeholder will be
                                replaced by the newely calculated gradients instead of accumulating the new gradients.
                                This can speed up the function runtime by around 10%.
        :return: A list containing the total loss and the individual network heads losses
        """

        if self.accumulated_gradients is None:
            self.reset_accumulated_gradients()

        # feed inputs
        if additional_fetches is None:
            additional_fetches = []
        feed_dict = self.create_feed_dict(inputs)

        # feed targets
        targets = force_list(targets)
        for placeholder_idx, target in enumerate(targets):
            feed_dict[self.targets[placeholder_idx]] = target

        # feed importance weights
        importance_weights = force_list(importance_weights)
        for placeholder_idx, target_ph in enumerate(targets):
            if len(importance_weights) <= placeholder_idx or importance_weights[placeholder_idx] is None:
                importance_weight = np.ones(target_ph.shape[0])
            else:
                importance_weight = importance_weights[placeholder_idx]
            importance_weight = np.reshape(importance_weight, (-1,) + (1,)*(len(target_ph.shape)-1))

            feed_dict[self.importance_weights[placeholder_idx]] = importance_weight

        if self.optimizer_type != 'LBFGS':

            # feed the lstm state if necessary
            if self.middleware.__class__.__name__ == 'LSTMMiddleware':
                # we can't always assume that we are starting from scratch here can we?
                feed_dict[self.middleware.c_in] = self.middleware.c_init
                feed_dict[self.middleware.h_in] = self.middleware.h_init

            fetches = self.train_fetches + additional_fetches
            if self.ap.visualization.tensorboard:
                fetches += [self.merged]

            # get grads
            result = self.sess.run(fetches, feed_dict=feed_dict)
            if hasattr(self, 'train_writer') and self.train_writer is not None:
                self.train_writer.add_summary(result[-1], self.sess.run(self.global_step))

            # extract the fetches
            norm_unclipped_grads, grads, total_loss, losses = result[:4]
            if self.middleware.__class__.__name__ == 'LSTMMiddleware':
                (self.curr_rnn_c_in, self.curr_rnn_h_in) = result[4]
            fetched_tensors = []
            if len(additional_fetches) > 0:
                fetched_tensors = result[self.additional_fetches_start_idx:self.additional_fetches_start_idx +
                                                                      len(additional_fetches)]

            # accumulate the gradients
            for idx, grad in enumerate(grads):
                if no_accumulation:
                    self.accumulated_gradients[idx] = grad
                else:
                    self.accumulated_gradients[idx] += grad

            return total_loss, losses, norm_unclipped_grads, fetched_tensors

        else:
            self.optimizer.minimize(session=self.sess, feed_dict=feed_dict)

            return [0]

    def create_feed_dict(self, inputs):
        feed_dict = {}
        for input_name, input_value in inputs.items():
            if isinstance(input_name, str):
                if input_name not in self.inputs:
                    raise ValueError((
                        'input name {input_name} was provided to create a feed '
                        'dictionary, but there is no placeholder with that name. '
                        'placeholder names available include: {placeholder_names}'
                    ).format(
                        input_name=input_name,
                        placeholder_names=', '.join(self.inputs.keys())
                    ))

                feed_dict[self.inputs[input_name]] = input_value
            elif isinstance(input_name, tf.Tensor) and input_name.op.type == 'Placeholder':
                feed_dict[input_name] = input_value
            else:
                raise ValueError((
                    'input dictionary expects strings or placeholders as keys, '
                    'but found key {key} of type {type}'
                ).format(
                    key=input_name,
                    type=type(input_name),
                ))

        return feed_dict

    def apply_and_reset_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights and resets the accumulation placeholder
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        """
        self.apply_gradients(gradients, scaler)
        self.reset_accumulated_gradients()

    def wait_for_all_workers_to_lock(self, lock: str, include_only_training_workers: bool=False):
        """
        Waits for all the workers to lock a certain lock and then continues
        :param lock: the name of the lock to use
        :param include_only_training_workers: wait only for training workers or for all the workers?
        :return: None
        """
        if include_only_training_workers:
            num_workers_to_wait_for = self.ap.task_parameters.num_training_tasks
        else:
            num_workers_to_wait_for = self.ap.task_parameters.num_tasks

        # lock
        if hasattr(self, '{}_counter'.format(lock)):
            self.sess.run(getattr(self, lock))
            while self.sess.run(getattr(self, '{}_counter'.format(lock))) % num_workers_to_wait_for != 0:
                time.sleep(0.00001)
            # self.sess.run(getattr(self, '{}_init'.format(lock)))
        else:
            raise ValueError("no counter was defined for the lock {}".format(lock))

    def wait_for_all_workers_barrier(self, include_only_training_workers: bool=False):
        """
        A barrier that allows waiting for all the workers to finish a certain block of commands
        :param include_only_training_workers: wait only for training workers or for all the workers?
        :return: None
        """
        self.wait_for_all_workers_to_lock('lock', include_only_training_workers=include_only_training_workers)
        self.sess.run(self.lock_init)

        # we need to lock again (on a different lock) in order to prevent a situation where one of the workers continue
        # and then was able to first increase the lock again by one, only to have a late worker to reset it again.
        # so we want to make sure that all workers are done resetting the lock before continuting to reuse that lock.

        self.wait_for_all_workers_to_lock('release', include_only_training_workers=include_only_training_workers)
        self.sess.run(self.release_init)

    def apply_gradients(self, gradients, scaler=1.):
        """
        Applies the given gradients to the network weights
        :param gradients: The gradients to use for the update
        :param scaler: A scaling factor that allows rescaling the gradients before applying them.
                       The gradients will be MULTIPLIED by this factor
        """
        if self.network_parameters.async_training or not isinstance(self.ap.task_parameters, DistributedTaskParameters):
            if hasattr(self, 'global_step') and not self.network_is_local:
                self.sess.run(self.inc_step)

        if self.optimizer_type != 'LBFGS':

            if self.distributed_training and not self.network_parameters.async_training:
                # rescale the gradients so that they average out with the gradients from the other workers
                if self.network_parameters.scale_down_gradients_by_number_of_workers_for_sync_training:
                    scaler /= float(self.ap.task_parameters.num_training_tasks)

            # rescale the gradients
            if scaler != 1.:
                for gradient in gradients:
                    gradient *= scaler

            # apply the gradients
            feed_dict = dict(zip(self.weights_placeholders, gradients))
            if self.distributed_training and self.network_parameters.shared_optimizer \
                    and not self.network_parameters.async_training:
                # synchronous distributed training with shared optimizer:
                # - each worker adds its gradients to the shared gradients accumulators
                # - we wait for all the workers to add their gradients
                # - the chief worker (worker with task index = 0) applies the gradients once and resets the accumulators

                self.sess.run(self.accumulate_shared_gradients, feed_dict=feed_dict)

                self.wait_for_all_workers_barrier(include_only_training_workers=True)

                if self.is_chief:
                    self.sess.run(self.update_weights_from_shared_gradients)
                    self.sess.run(self.init_shared_accumulated_gradients)
            else:
                # async distributed training / distributed training with independent optimizer
                #  / non-distributed training - just apply the gradients
                feed_dict = dict(zip(self.weights_placeholders, gradients))
                self.sess.run(self.update_weights_from_batch_gradients, feed_dict=feed_dict)

            # release barrier
            if self.distributed_training and not self.network_parameters.async_training:
                self.wait_for_all_workers_barrier(include_only_training_workers=True)

    def predict(self, inputs, outputs=None, squeeze_output=True, initial_feed_dict=None):
        """
        Run a forward pass of the network using the given input
        :param inputs: The input for the network
        :param outputs: The output for the network, defaults to self.outputs
        :param squeeze_output: call squeeze_list on output
        :param initial_feed_dict: a dictionary to use as the initial feed_dict. other inputs will be added to this dict
        :return: The network output

        WARNING: must only call once per state since each call is assumed by LSTM to be a new time step.
        """
        feed_dict = self.create_feed_dict(inputs)
        if initial_feed_dict:
            feed_dict.update(initial_feed_dict)
        if outputs is None:
            outputs = self.outputs

        if self.middleware.__class__.__name__ == 'LSTMMiddleware':
            feed_dict[self.middleware.c_in] = self.curr_rnn_c_in
            feed_dict[self.middleware.h_in] = self.curr_rnn_h_in

            output, (self.curr_rnn_c_in, self.curr_rnn_h_in) = self.sess.run([outputs, self.middleware.state_out],
                                                                             feed_dict=feed_dict)
        else:
            output = self.sess.run(outputs, feed_dict)

        if squeeze_output:
            output = squeeze_list(output)
        return output

    @staticmethod
    def parallel_predict(sess: Any,
                         network_input_tuples: List[Tuple['TensorFlowArchitecture', Dict[str, np.ndarray]]]) ->\
            List[np.ndarray]:
        """
        :param sess: active session to use for prediction
        :param network_input_tuples: tuple of network and corresponding input
        :return: list of outputs from all networks
        """
        feed_dict = {}
        fetches = []

        for network, input in network_input_tuples:
            feed_dict.update(network.create_feed_dict(input))
            fetches += network.outputs

        outputs = sess.run(fetches, feed_dict)

        return outputs

    def train_on_batch(self, inputs, targets, scaler=1., additional_fetches=None, importance_weights=None):
        """
        Given a batch of examples and targets, runs a forward pass & backward pass and then applies the gradients
        :param additional_fetches: Optional tensors to fetch during the training process
        :param inputs: The input for the network
        :param targets: The targets corresponding to the input batch
        :param scaler: A scaling factor that allows rescaling the gradients before applying them
        :param importance_weights: A coefficient for each sample in the batch, which will be used to rescale the loss
                                   error of this sample. If it is not given, the samples losses won't be scaled
        :return: The loss of the network
        """
        if additional_fetches is None:
            additional_fetches = []
        force_list(additional_fetches)
        loss = self.accumulate_gradients(inputs, targets, additional_fetches=additional_fetches,
                                         importance_weights=importance_weights)
        self.apply_and_reset_gradients(self.accumulated_gradients, scaler)
        return loss

    def get_weights(self):
        """
        :return: a list of tensors containing the network weights for each layer
        """
        return self.weights

    def set_weights(self, weights, new_rate=1.0):
        """
        Sets the network weights from the given list of weights tensors
        """
        feed_dict = {}
        old_weights, new_weights = self.sess.run([self.get_weights(), weights])
        for placeholder_idx, new_weight in enumerate(new_weights):
            feed_dict[self.weights_placeholders[placeholder_idx]]\
                = new_rate * new_weight + (1 - new_rate) * old_weights[placeholder_idx]
        self.sess.run(self.update_weights_from_list, feed_dict)

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

    def set_is_training(self, state: bool):
        """
        Set the phase of the network between training and testing
        :param state: The current state (True = Training, False = Testing)
        :return: None
        """
        self.set_variable_value(self.assign_is_training, state, self.is_training_placeholder)

    def reset_internal_memory(self):
        """
        Reset any internal memory used by the network. For example, an LSTM internal state
        :return: None
        """
        # initialize LSTM hidden states
        if self.middleware.__class__.__name__ == 'LSTMMiddleware':
            self.curr_rnn_c_in = self.middleware.c_init
            self.curr_rnn_h_in = self.middleware.h_init

    def collect_savers(self, parent_path_suffix: str) -> SaverCollection:
        """
        Collection of all checkpoints for the network (typically only one checkpoint)
        :param parent_path_suffix: path suffix of the parent of the network
            (e.g. could be name of level manager plus name of agent)
        :return: checkpoint collection for the network
        """
        savers = SaverCollection()
        if not self.distributed_training:
            savers.add(GlobalVariableSaver(self.name))
        return savers


def save_onnx_graph(input_nodes, output_nodes, checkpoint_save_dir: str) -> None:
    """
    Given the input nodes and output nodes of the TF graph, save it as an onnx graph
    This requires the TF graph and the weights checkpoint to be stored in the experiment directory.
    It then freezes the graph (merging the graph and weights checkpoint), and converts it to ONNX.

    :param input_nodes: A list of input nodes for the TF graph
    :param output_nodes: A list of output nodes for the TF graph
    :param checkpoint_save_dir: The directory to save the ONNX graph to
    :return: None
    """
    import tf2onnx  # just to verify that tf2onnx is installed

    # freeze graph
    frozen_graph_path = os.path.join(checkpoint_save_dir, "frozen_graph.pb")
    freeze_graph_command = [
        "python -m tensorflow.python.tools.freeze_graph",
        "--input_graph={}".format(os.path.join(checkpoint_save_dir, "graphdef.pb")),
        "--input_binary=true",
        "--output_node_names='{}'".format(','.join([o.split(":")[0] for o in output_nodes])),
        "--input_checkpoint={}".format(tf.train.latest_checkpoint(checkpoint_save_dir)),
        "--output_graph={}".format(frozen_graph_path)
    ]
    start_shell_command_and_wait(" ".join(freeze_graph_command))

    # convert graph to onnx
    onnx_graph_path = os.path.join(checkpoint_save_dir, "model.onnx")
    convert_to_onnx_command = [
        "python -m tf2onnx.convert",
        "--input {}".format(frozen_graph_path),
        "--inputs '{}'".format(','.join(input_nodes)),
        "--outputs '{}'".format(','.join(output_nodes)),
        "--output {}".format(onnx_graph_path),
        "--verbose"
    ]
    start_shell_command_and_wait(" ".join(convert_to_onnx_command))
