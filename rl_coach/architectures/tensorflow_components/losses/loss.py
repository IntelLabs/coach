from tensorflow import keras


class GeneralLoss(keras.losses.Loss):
    def __init__(self, loss_type='MeanSquaredError', **kwargs):
        self.loss_type = loss_type
        self.loss_fn = keras.losses.get(self.loss_type)
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "loss_type": self.loss_type}


class Loss(keras.losses.Loss):

    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    # def _loss_output(self, outputs):
    #     return tuple(o[0] for o in outputs)