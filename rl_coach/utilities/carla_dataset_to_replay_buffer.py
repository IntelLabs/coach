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
import os
import sys

import h5py
import numpy as np

from rl_coach.core_types import Transition
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplay
from rl_coach.utils import ProgressBar, start_shell_command_and_wait
from rl_coach.logger import screen


def maybe_download(dataset_root):
    if not dataset_root or not os.path.exists(os.path.join(dataset_root, "AgentHuman")):
        screen.log_title("Downloading the CARLA dataset. This might take a while.")

        google_drive_download_id = "1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY"
        filename_to_save = "datasets/CORL2017ImitationLearningData.tar.gz"
        download_command = 'wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=' \
                           '$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies ' \
                           '--no-check-certificate \"https://docs.google.com/uc?export=download&id={}\" -O- | ' \
                           'sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={}" -O {} && rm -rf /tmp/cookies.txt'\
                           .format(google_drive_download_id, google_drive_download_id, filename_to_save)

        # start downloading and wait for it to finish
        start_shell_command_and_wait(download_command)

        screen.log_title("Unzipping the dataset")
        unzip_command = 'tar -xzf {} --checkpoint=.10000'.format(filename_to_save)
        if dataset_root is not None:
            unzip_command += " -C {}".format(dataset_root)

        if not os.path.exists(dataset_root):
            os.makedirs(dataset_root)
        start_shell_command_and_wait(unzip_command)


def create_dataset(dataset_root, output_path):
    maybe_download(dataset_root)

    dataset_root = os.path.join(dataset_root, 'AgentHuman')
    train_set_root = os.path.join(dataset_root, 'SeqTrain')
    validation_set_root = os.path.join(dataset_root, 'SeqVal')

    # training set extraction
    memory = ExperienceReplay(max_size=(MemoryGranularity.Transitions, sys.maxsize))
    train_set_files = sorted(os.listdir(train_set_root))
    print("found {} files".format(len(train_set_files)))
    progress_bar = ProgressBar(len(train_set_files))
    for file_idx, file in enumerate(train_set_files[:3000]):
        progress_bar.update(file_idx, "extracting file {}".format(file))
        train_set = h5py.File(os.path.join(train_set_root, file), 'r')
        observations = train_set['rgb'][:]  # forward camera
        measurements = np.expand_dims(train_set['targets'][:, 10], -1)  # forward speed
        actions = train_set['targets'][:, :3]  # steer, gas, brake

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
    print("Saving pickle file to {}".format(output_path))
    memory.save(output_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-d', '--dataset_root', help='The path to the CARLA dataset root folder')
    argparser.add_argument('-o', '--output_path', help='The path to save the resulting replay buffer',
                           default='carla_train_set_replay_buffer.p')
    args = argparser.parse_args()

    create_dataset(args.dataset_root, args.output_path)
