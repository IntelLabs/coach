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


class HeadLoss(keras.losses.Loss):
    """
    ABC for loss functions of each head. Child class must implement input_schema() and loss_forward()
    """
    def __init__(self, *args, **kwargs):
        super(HeadLoss, self).__init__(*args, **kwargs)

    def _loss_output(self, outputs):
        """
        Saves the returned output as the schema and returns output values in a list
        :return: list of output values
        """
        output_schema = [o[1] for o in outputs]
        assert self._output_schema is None or self._output_schema == output_schema
        self._output_schema = output_schema
        return tuple(o[0] for o in outputs)