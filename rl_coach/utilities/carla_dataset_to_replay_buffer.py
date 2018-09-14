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

import argparse

import h5py
import os
import sys
import numpy as np
from rl_coach.utils import ProgressBar
from rl_coach.core_types import Transition
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplay


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-d', '--dataset_root', help='The path to the CARLA dataset root folder')
    argparser.add_argument('-o', '--output_path', help='The path to save the resulting replay buffer',
                           default='carla_train_set_replay_buffer.p')
    args = argparser.parse_args()

    train_set_root = os.path.join(args.dataset_root, 'SeqTrain')
    validation_set_root = os.path.join(args.dataset_root, 'SeqVal')

    # training set extraction
    memory = ExperienceReplay(max_size=(MemoryGranularity.Transitions, sys.maxsize))
    train_set_files = sorted(os.listdir(train_set_root))
    print("found {} files".format(len(train_set_files)))
    progress_bar = ProgressBar(len(train_set_files))
    for file_idx, file in enumerate(train_set_files[:3000]):
        progress_bar.update(file_idx, "extracting file {}".format(file))
        train_set = h5py.File(os.path.join(train_set_root, file), 'r')
        observations = train_set['rgb'][:]                                   # forward camera
        measurements = np.expand_dims(train_set['targets'][:, 10], -1)       # forward speed
        actions = train_set['targets'][:, :3]                                # steer, gas, brake
        actions[:, 1] -= actions[:, 2]
        actions = actions[:, :2][:, ::-1]

        high_level_commands = train_set['targets'][:, 24].astype('int') - 2  # follow lane, left, right, straight

        file_length = train_set['rgb'].len()
        assert train_set['rgb'].len() == train_set['targets'].len()

        for transition_idx in range(file_length):
            transition = Transition(
                state={
                    'CameraRGB': observations[transition_idx],
                    'measurements': measurements[transition_idx],
                    'high_level_command': high_level_commands[transition_idx]
                },
                action=actions[transition_idx],
                reward=0
            )
            memory.store(transition)
    progress_bar.close()
    print("Saving pickle file to {}".format(args.output_path))
    memory.save(args.output_path)
