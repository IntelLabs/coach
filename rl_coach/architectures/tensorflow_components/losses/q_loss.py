
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError

from rl_coach.architectures.tensorflow_components.losses.head_loss import HeadLoss, LossInputSchema

class QLoss(HeadLoss):
    def __init__(self, network_name,
                 head_idx: int = 0,
                 loss_type: Loss = MeanSquaredError,
                 loss_weight=1.0,
                 **kwargs):
        """
        Loss for Q-Value Head.

        :param loss_type: loss function with default of mean squared error (i.e. L2Loss).
        :param loss_weight: scalar used to adjust relative weight of loss (if using this loss with others).

        :param batch_axis: axis used for mini-batch (default is 0) and excluded from loss aggregation.
        """
        super().__init__(**kwargs)

        @property
        def input_schema(self) -> LossInputSchema:
            return LossInputSchema(
                head_outputs=['pred'],
                agent_inputs=[],
                targets=['target']
            )

        assert (loss_type == MeanSquaredError) or (loss_type == Huber), "Only expecting L2Loss or HuberLoss."
        self.loss_type = loss_type
        self.loss_fn = keras.losses.mean_squared_error#keras.losses.get(loss_type)
        # sample_weight can be used like https://github.com/keras-team/keras/blob/master/keras/losses.py


    def call(self, target, prediction):
        """
        Used for forward pass through loss computations.
        :param prediction: state-action q-values predicted by QHead network, of shape (batch_size, num_actions).
        :param target: actual state-action q-values, of shape (batch_size, num_actions).
        :return: loss, of shape (batch_size).
        """
        # TODO: preferable to return a tensor containing one loss per instance, rather than returning the mean loss.
        #  This way, Keras can apply class weights or sample weights when requested.
        loss = tf.reduce_mean(self.loss_fn(prediction, target))
        return loss

