
from rl_coach.architectures.tensorflow_components.losses.loss import HeadLoss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError


from rl_coach.architectures.tensorflow_components.losses.loss import HeadLoss

class QHeadLoss(HeadLoss):
    def __init__(self, loss_type: Loss = MeanSquaredError, **kwargs):
        """
        Loss for Q-Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param weight: scalar used to adjust relative weight of loss (if using this loss with others).
        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super().__init__(**kwargs)

        self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)


    def call(self, y_true, y_pred):
        """
        Used for forward pass through loss computations.
        :param pred: state-action q-values predicted by QHead network, of shape (batch_size, num_actions).
        :param target: actual state-action q-values, of shape (batch_size, num_actions).
        :return: loss, of shape (batch_size).
        """
        # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
        #  This way, Keras can apply class weights or sample weights when requested.
        loss = tf.reduce_mean(self.loss_fn(y_pred, y_true))
        return loss
        #return [(loss, LOSS_OUT_TYPE_LOSS)]
