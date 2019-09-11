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
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError

from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS

class VLoss(keras.losses.Loss):

    def __init__(self, network_name,
                 head_idx: int = 0,
                 loss_type: Loss = MeanSquaredError,
                 loss_weight=1.0,
                 **kwargs):
        """
        Loss for Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)

    def call(self, prediction, target):
        """
        Used for forward pass through loss computations.

        :param prediction: state values predicted by VHead network, of shape (batch_size).
        :param target: actual state values, of shape (batch_size).
        :return: loss, of shape (batch_size).
        """
        # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
        #  This way, Keras can apply class weights or sample weights when requested.
        loss = tf.reduce_mean(self.loss_fn(prediction, target))
        return loss
        # loss = self.loss_fn(pred, target).mean()
        # return [(loss, LOSS_OUT_TYPE_LOSS)]