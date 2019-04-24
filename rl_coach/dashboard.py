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

"""
To run Coach Dashboard, run the following command:
python3 dashboard.py
"""

import sys
sys.path.append('.')

import os

from rl_coach.dashboard_components.experiment_board import display_directory_group, display_files
from rl_coach.dashboard_components.globals import doc
import rl_coach.dashboard_components.boards
from rl_coach.dashboard_components.landing_page import landing_page

doc.add_root(landing_page)

import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--experiment_dir',
                    help="(string) The path of an experiment dir to open",
                    default=None,
                    type=str)
parser.add_argument('-f', '--experiment_files',
                    help="(string) The path of an experiment file to open",
                    default=None,
                    type=str)
parser.add_argument('-rc', '--allow_remote_connection',
                    help="(flag) Allow remote connection to view dashboard results from a remote machine. "
                         "Note this opens up the connection and allows anyone who tries to connect to the "
                         "ip address/port to connect and view the bokeh results via their browser.",
                    action='store_true')

args = parser.parse_args()

if args.experiment_dir:
    doc.add_timeout_callback(lambda: display_directory_group(args.experiment_dir), 1000)
elif args.experiment_files:
    files = []
    for file_pattern in args.experiment_files:
        files.extend(glob.glob(args.experiment_files))
    doc.add_timeout_callback(lambda: display_files(files), 1000)


def main():
    from rl_coach.utils import get_open_port

    dashboard_path = os.path.realpath(__file__)
    port = get_open_port()
    command = 'bokeh serve --show {path} --port {port}'.format(path=dashboard_path, port=port)

    if args.allow_remote_connection:
        # when allowing remote connection, selecting an experiment or a file via the GUI buttons do not seem to work
        # well from remote. Instead, we only allow entering an experiment dir from command line.
        if not args.experiment_dir and not args.experiment_files:
            raise ValueError("The allow_remote_connection flag only works in conjunction with either the experiment_dir"
                             " or the experiment_files flag. ")

        # allow-websocket-origin = * allows connections from a remote machine.
        command += ' --allow-websocket-origin=*'

    if args.experiment_dir or args.experiment_files:
        command += ' --args'
        if args.experiment_dir:
            command += ' --experiment_dir {}'.format(args.experiment_dir)
        if args.experiment_files:
            command += ' --experiment_files {}'.format(args.experiment_files)

    os.system(command)


if __name__ == "__main__":
    main()
