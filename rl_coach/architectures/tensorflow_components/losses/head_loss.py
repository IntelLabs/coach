from tensorflow import keras
from typing import Dict, List, Union, Tuple
from tensorflow import Tensor
import numpy as np
import inspect

LOSS_OUT_TYPE_LOSS = 'loss'
LOSS_OUT_TYPE_REGULARIZATION = 'regularization'


class LossInputSchema(object):
    """
    Helper class to contain schema for loss hybrid_forward input
    """
    def __init__(self, head_outputs: List[str], agent_inputs: List[str], targets: List[str]):
        """
        :param head_outputs: list of argument names in call that are the outputs of the head.
            The order and number MUST MATCH the output from the head.
        :param agent_inputs: list of argument names in hybrid_forward that are inputs from the agent.
            The order and number MUST MATCH `output_<head_type_idx>_<order>` for this head.
        :param targets: list of argument names in hybrid_forward that are targets for the loss.
            The order and number MUST MATCH targets passed from the agent.
        """
        self._head_outputs = head_outputs
        self._agent_inputs = agent_inputs
        self._targets = targets

    @property
    def head_outputs(self):
        return self._head_outputs

    @property
    def agent_inputs(self):
        return self._agent_inputs

    @property
    def targets(self):
        return self._targets


#class HeadLoss(keras.losses.Loss):
class HeadLoss(keras.layers.Layer):
    """
    ABC for loss functions of each agent. Child class must implement input_schema() and loss_forward()
    """

    def __init__(self, *args, **kwargs):
        super(HeadLoss, self).__init__(*args, **kwargs)
        self._output_schema = None  # type: List[str]

    @property
    def input_schema(self) -> LossInputSchema:
        """
        :return: schema for input of hybrid_forward. Read docstring for LossInputSchema for details.
        """
        raise NotImplementedError

    @property
    def output_schema(self) -> List[str]:
        """
        :return: schema for output of hybrid_forward. Must contain 'loss' and 'regularization' keys at least once.
            The order and total number must match that of returned values from the loss. 'loss' and 'regularization'
            are special keys. Any other string is treated as auxiliary outputs and must include match auxiliary
            fetch names returned by the head.
        """
        return self._output_schema

    def forward(self, *args):
        """
        Override forward() so that number of outputs can be checked against the schema
        """

        outputs = super(HeadLoss, self).forward(*args)

        if isinstance(outputs, tuple) or isinstance(outputs, list):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, NDArray) or isinstance(outputs, Symbol)
            num_outputs = 1
        assert num_outputs == len(self.output_schema), "Number of outputs don't match schema ({} != {})".format(
            num_outputs, len(self.output_schema))
        return outputs

    def _loss_output(self, outputs: List[Tuple[Tensor, str]]):
        """
        Must be called on the output from call ().
        Saves the returned output as the schema and returns output values in a list
        :return: list of output values
        """
        output_schema = [o[1] for o in outputs]
        assert self._output_schema is None or self._output_schema == output_schema
        self._output_schema = output_schema
        return tuple(o[0] for o in outputs)

    def call(self, head_output, agent_input, target):
        return self._loss_output(self.loss_forward(head_output, agent_input, target))

    def loss_forward(self, head_output, agent_input, target):
        """
        Similar to hybrid_forward, but returns list of (NDArray, type_str)
        """
        raise NotImplementedError

    def align_loss_args(self,
            head_outputs: List[Tensor],
            agent_inputs: List[np.ndarray],
            targets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Creates a list of arguments from head_outputs, agent_inputs, and targets aligned with parameters of
        loss.loss_forward() based on their name in loss input_schema
        :param head_outputs: list of all head_outputs for this loss
        :param agent_inputs: list of all agent_inputs for this loss
        :param targets: list of all targets for this loss
        :return: list of arguments in correct order to be passed to loss
        """
        arg_list = list()
        schema = self.input_schema
        assert len(schema.head_outputs) == len(head_outputs)
        assert len(schema.agent_inputs) == len(agent_inputs)
        assert len(schema.targets) == len(targets)

        prev_found = True
        for arg_name in inspect.getfullargspec(self.loss_forward).args[2:]:  # First two args are self and F
            found = False
            for schema_list, data in [(schema.head_outputs, head_outputs),
                                      (schema.agent_inputs, agent_inputs),
                                      (schema.targets, targets)]:
                try:
                    arg_list.append(data[schema_list.index(arg_name)])
                    found = True
                    break
                except ValueError:
                    continue
            assert not found or prev_found, "missing arguments detected!"
            prev_found = found
        return arg_list





