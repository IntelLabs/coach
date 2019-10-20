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
from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema

from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS

class VLoss(HeadLoss):

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
        assert (loss_type == MeanSquaredError) or (loss_type == Huber), "Only expecting L2Loss or HuberLoss."
        self.loss_type = loss_type
        #self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @property
    def input_schema(self) -> LossInputSchema:
        return LossInputSchema(
            head_outputs=['value_prediction'],
            agent_inputs=[],
            targets=['target']
        )

    def loss_forward(self, value_prediction, target):
        loss = self.loss_fn(value_prediction, target)
        return [(loss, LOSS_OUT_TYPE_LOSS)]
