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


import os
from os.path import basename

import pandas as pd
from pandas.errors import EmptyDataError

from rl_coach.dashboard_components.signals_file_base import SignalsFileBase
from rl_coach.dashboard_components.globals import x_axis_options
from rl_coach.utils import break_file_path


class SignalsFile(SignalsFileBase):
    def __init__(self, csv_path, load=True, plot=None, use_dir_name=False):
        super().__init__(plot)
        self.use_dir_name = use_dir_name
        self.full_csv_path = csv_path
        self.dir, self.filename, _ = break_file_path(csv_path)

        if use_dir_name:
            parent_directory_path = os.path.abspath(os.path.join(os.path.dirname(csv_path), '..'))
            if len(os.listdir(parent_directory_path)) == 1:
                # get the parent directory name (since the current directory is the timestamp directory)
                self.dir = parent_directory_path
                self.filename = basename(self.dir)
            else:
                # get the common directory for all the experiments
                self.dir = os.path.dirname(csv_path)
                self.filename = "{}/{}".format(basename(parent_directory_path), basename(self.dir))

        if load:
            self.load()
            # this helps set the correct x axis
            self.change_averaging_window(1, force=True)

    def load_csv(self, idx=None, result=None):
        # load csv and fix sparse data.
        # csv can be in the middle of being written so we use try - except
        new_csv = None
        while new_csv is None:
            try:
                new_csv = pd.read_csv(self.full_csv_path)
                break
            except EmptyDataError:
                new_csv = None
                continue

        new_csv['Wall-Clock Time'] /= 60.
        new_csv = new_csv.interpolate()
        # remove signals which don't contain any values
        for k, v in new_csv.isna().all().items():
            if v and k not in x_axis_options:
                del new_csv[k]
        new_csv.fillna(value=0, inplace=True)

        self.csv = new_csv

        self.last_modified = os.path.getmtime(self.full_csv_path)

        if idx is not None:
            result[idx] = (self.csv, self.last_modified)

    def file_was_modified_on_disk(self):
        return self.last_modified != os.path.getmtime(self.full_csv_path)
