# Copyright (c) 2019 Intel Corporation
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
from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema, LOSS_OUT_TYPE_LOSS


class VLoss(HeadLoss):

    def __init__(self,
                 network_name,
                 head_idx: int = 0,
                 loss_type: Loss = MeanSquaredError,
                 loss_weight: float=1.):
        """
        Loss for Value Head.
        :param head_idx: the index of the corresponding head.
        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).
        """
        super(VLoss, self).__init__(name=network_name)
        self.head_idx = head_idx
        assert (loss_type == MeanSquaredError) or (loss_type == Huber), "Only expecting L2Loss or HuberLoss."
        self.loss_fn = keras.losses.get(loss_type)()
        #self.loss_fn = tf.keras.losses.MeanSquaredError()

    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            model_outputs=['value_prediction'],
            non_trainable_args=['target']
        )

    def loss_forward(self, value_prediction, target):
        """
        Used for forward pass through loss computations.
        :param value_prediction: state values predicted by VHead network, of shape (batch_size).
        :param target: actual state values, of shape (batch_size).
        :return: loss, of shape (batch_size).
        """
        loss = self.loss_fn(value_prediction, target)
        return {LOSS_OUT_TYPE_LOSS: [loss]}
