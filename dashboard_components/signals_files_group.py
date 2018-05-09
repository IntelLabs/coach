import os
from os.path import basename

import pandas as pd

from dashboard_components.globals import x_axis_options, add_directory_csv_files, show_spinner
from dashboard_components.signals_file import SignalsFile
from dashboard_components.signals_file_base import SignalsFileBase


class SignalsFilesGroup(SignalsFileBase):
    def __init__(self, csv_paths, plot=None):
        super().__init__(plot)
        self.full_csv_paths = csv_paths
        self.signals_files = []
        if len(csv_paths) == 1 and os.path.isdir(csv_paths[0]):
            self.signals_files = [SignalsFile(str(file), load=False, plot=plot) for file in add_directory_csv_files(csv_paths[0])]
        else:
            for csv_path in csv_paths:
                if os.path.isdir(csv_path):
                    self.signals_files.append(SignalsFilesGroup(add_directory_csv_files(csv_path), plot=plot))
                else:
                    self.signals_files.append(SignalsFile(str(csv_path), load=False, plot=plot))
        parent_directory_path = os.path.abspath(os.path.join(os.path.dirname(csv_paths[0]), '..'))

        if len(csv_paths) == 1 and len(os.listdir(parent_directory_path)) == 1:
            # get the parent directory name (since the current directory is the timestamp directory)
            self.dir = os.path.abspath(os.path.join(os.path.dirname(csv_paths[0]), '..'))
        else:
            # get the common directory for all the experiments
            self.dir = os.path.dirname(os.path.commonprefix(csv_paths))

        self.filename = '{} - Group({})'.format(basename(self.dir), len(self.signals_files))

        self.load()

    def load_csv(self):
        corrupted_files_idx = []
        for idx, signal_file in enumerate(self.signals_files):
            signal_file.load_csv()
            if not all(option in signal_file.csv.keys() for option in x_axis_options):
                print("Warning: {} file seems to be corrupted and does contain the necessary columns "
                      "and will not be rendered".format(signal_file.filename))
                corrupted_files_idx.append(idx)

        for file_idx in corrupted_files_idx:
            del self.signals_files[file_idx]

        # get the stats of all the columns
        if len(self.signals_files) > 1:
            csv_group = pd.concat([signals_file.csv for signals_file in self.signals_files])
            columns_to_remove = [s for s in csv_group.columns if '/Stdev' in s] + \
                                [s for s in csv_group.columns if '/Min' in s] + \
                                [s for s in csv_group.columns if '/Max' in s]
            for col in columns_to_remove:
                del csv_group[col]
            csv_group = csv_group.groupby(csv_group.index)
            self.csv_mean = csv_group.mean()
            self.csv_mean.columns = [s + '/Mean' for s in self.csv_mean.columns]
            self.csv_stdev = csv_group.std()
            self.csv_stdev.columns = [s + '/Stdev' for s in self.csv_stdev.columns]
            self.csv_min = csv_group.min()
            self.csv_min.columns = [s + '/Min' for s in self.csv_min.columns]
            self.csv_max = csv_group.max()
            self.csv_max.columns = [s + '/Max' for s in self.csv_max.columns]

            # get the indices from the file with the least number of indices and which is not an evaluation worker
            file_with_min_indices = self.signals_files[0]
            for signals_file in self.signals_files:
                if signals_file.csv.shape[0] < file_with_min_indices.csv.shape[0] and \
                                'Training reward' in signals_file.csv.keys():
                    file_with_min_indices = signals_file
            self.index_columns = file_with_min_indices.csv[x_axis_options]

            # concat the stats and the indices columns
            num_rows = file_with_min_indices.csv.shape[0]
            self.csv = pd.concat([self.index_columns, self.csv_mean.head(num_rows), self.csv_stdev.head(num_rows),
                                  self.csv_min.head(num_rows), self.csv_max.head(num_rows)], axis=1)

            # remove the stat columns for the indices columns
            columns_to_remove = [s + '/Mean' for s in x_axis_options] + \
                                [s + '/Stdev' for s in x_axis_options] + \
                                [s + '/Min' for s in x_axis_options] + \
                                [s + '/Max' for s in x_axis_options]
            for col in columns_to_remove:
                del self.csv[col]
        else:  # This is a group of a single file
            self.csv = self.signals_files[0].csv

        # # convert wall clock time to minutes - isn't needed because the sub-signals are already scaled
        # self.csv['Wall-Clock Time'] /= 60.

        # remove NaNs
        self.csv.fillna(value=0, inplace=True)  # removing this line will make bollinger bands fail
        for key in self.csv.keys():
            if 'Stdev' in key and 'Evaluation' not in key:
                self.csv[key] = self.csv[key].fillna(value=0)

        for signal_file in self.signals_files:
            signal_file.update_source_and_signals()

    def reload_data(self):
        for signal_file in self.signals_files:
            signal_file.reload_data()
        SignalsFileBase.reload_data(self)

    def update_x_axis_index(self):
        for signal_file in self.signals_files:
            signal_file.update_x_axis_index()
        SignalsFileBase.update_x_axis_index(self)

    def toggle_y_axis(self, signal_name=None):
        for signal in self.signals.values():
            if signal.selected:
                signal.toggle_axis()
                for signal_file in self.signals_files:
                    signal_file.toggle_y_axis(signal.name)

    def change_averaging_window(self, new_size, force=False, signals=None):
        for signal_file in self.signals_files:
            signal_file.change_averaging_window(new_size, force, signals)
        SignalsFileBase.change_averaging_window(self, new_size, force, signals)

    def set_signal_selection(self, signal_name, val):
        self.show_files_separately(self.separate_files)
        SignalsFileBase.set_signal_selection(self, signal_name, val)

    def file_was_modified_on_disk(self):
        for signal_file in self.signals_files:
            if signal_file.file_was_modified_on_disk():
                return True
        return False

    def show_files_separately(self, val):
        self.separate_files = val
        for signal in self.signals.values():
            if signal.selected:
                if val:
                    signal.set_dash("4 4")
                else:
                    signal.set_dash("")
            for signal_file in self.signals_files:
                try:
                    if val:
                        signal_file.set_signal_selection(signal.name, signal.selected)
                    else:
                        signal_file.set_signal_selection(signal.name, False)
                except:
                    pass