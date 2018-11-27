#
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


from rl_coach.core_types import Transition


class TransitionCollection(object):
    """
    Simple python implementation of transitions collection non-episodic memories
    are constructed on top of.
    """
    def __init__(self):
        super(TransitionCollection, self).__init__()

    def append(self, transition):
        pass

    def extend(self, transitions):
        for transition in transitions:
            self.append(transition)

    def __len__(self):
        pass

    def __del__(self, range: slice):
        # NOTE: the only slice used is the form: slice(None, n)
        # NOTE: if it is easier, what we really want here is the ability to
        # constrain the size of the collection. as new transitions are added,
        # old transitions can be removed to maintain a maximum collection size.
        pass

    def __getitem__(self, key: int):
        # NOTE: we can switch to a method which fetches multiple items at a time
        # if that would significantly improve performance
        pass

    def __iter__(self):
        # this is not high priority
        pass
