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


from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError

from rl_coach.architectures.mxnet_components.heads.head import LOSS_OUT_TYPE_LOSS





class VLoss(keras.losses.Loss):
    def __init__(self, loss_type: Loss = MeanSquaredError,
                 weight: float = 1.0,
                 batch_axis: int = 0) -> None:
        """
        Loss for Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super(VLoss, self).__init__(weight=weight, batch_axis=batch_axis)
        self.loss_fn = loss_type(weight=weight, batch_axis=batch_axis)



    def loss_forward(self, pred, target):
        """
        Used for forward pass through loss computations.

        :param F: backend api, either `mxnet.nd` or `mxnet.sym` (if block has been hybridized).
        :param pred: state values predicted by VHead network, of shape (batch_size).
        :param target: actual state values, of shape (batch_size).
        :return: loss, of shape (batch_size).
        """
        loss = self.loss_fn(pred, target).mean()
        return [(loss, LOSS_OUT_TYPE_LOSS)]