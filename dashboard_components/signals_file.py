import os

import pandas as pd
from pandas.errors import EmptyDataError

from dashboard_components.signals_file_base import SignalsFileBase
from utils import break_file_path


class SignalsFile(SignalsFileBase):
    def __init__(self, csv_path, load=True, plot=None):
        super().__init__(plot)
        self.full_csv_path = csv_path
        self.dir, self.filename, _ = break_file_path(csv_path)
        if load:
            self.load()
            # this helps set the correct x axis
            self.change_averaging_window(1, force=True)

    def load_csv(self):
        # load csv and fix sparse data.
        # csv can be in the middle of being written so we use try - except
        self.csv = None
        while self.csv is None:
            try:
                self.csv = pd.read_csv(self.full_csv_path)
                break
            except EmptyDataError:
                self.csv = None
                continue
        self.csv = self.csv.interpolate()
        self.csv.fillna(value=0, inplace=True)

        self.csv['Wall-Clock Time'] /= 60.

        self.last_modified = os.path.getmtime(self.full_csv_path)

    def file_was_modified_on_disk(self):
        return self.last_modified != os.path.getmtime(self.full_csv_path)