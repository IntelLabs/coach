#
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


from typing import Any, List
from rl_coach.saver import Saver


class TfSaver(Saver):
    """
    child class that implements saver for saving tensorflow DNN model.
    """

    def __init__(self,
                 name: str,
                 model):
        self._name = name
        self.model = model
        #self.model._set_inputs(inputs)
        self._weights_dict = model.get_weights()

    @property
    def path(self):
        """
        Relative path for save/load. If two checkpoint objects return the same path, they must be merge-able.
        """
        return ""  # use empty string for global file

    def save(self, sess: None, save_path: str) -> List[str]:
        """
        Save to save_path
        :param sess: active session for session-based frameworks (TF1 legacy). Must be Non
        :param save_path: full path to save checkpoint (typically directory plus checkpoint prefix plus self.path)
        :return: list of all saved paths
        """
        assert sess is None
        #self.model.save(save_path, save_format="tf")
        self.model.save_weights(save_path)

        # # Save the model weights
        # model_weights_path = "{}.{}.h5".format(save_path, 'weights')
        # self.model.save_weights(model_weights_path)
        #
        # # Save the model architecture
        # model_architecture_path = "{}.{}.json".format(save_path, 'architecture')
        # with open(model_architecture_path, 'w') as f:
        #     f.write(self.model.to_json())

        return [save_path]

    def restore(self, sess: Any, restore_path: str):
        """
        Restore from restore_path
        :param sess: active session for session-based frameworks (e.g. TF)
        :param restore_path: full path to load checkpoint from.
        """
        assert sess is None
        self.model.load_weights(restore_path)
        self._weights_dict = self.model.get_weights()

    def merge(self, other: "Saver"):
        """
        Merge other saver into this saver
        :param other: saver to be merged into self
        """
        pass




