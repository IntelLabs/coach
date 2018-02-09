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

from utils import *
import os
import datetime

import sys
import wx
import random
import pandas as pd
from pandas.io.common import EmptyDataError
import numpy as np
import colorsys
from bokeh.palettes import Dark2
from bokeh.layouts import row, column, widgetbox, Spacer
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, HoverTool, WheelZoomTool, PanTool, Legend
from bokeh.models.widgets import RadioButtonGroup, MultiSelect, Button, Select, Slider, Div, CheckboxGroup
from bokeh.models.glyphs import Patch
from bokeh.plotting import figure, show, curdoc
from utils import force_list
from utils import squeeze_list
from itertools import cycle
from os import listdir
from os.path import isfile, join, isdir, basename
from enum import Enum


class DialogApp(wx.App):
    def getFileDialog(self):
        with wx.FileDialog(None, "Open CSV file", wildcard="CSV files (*.csv)|*.csv",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_CHANGE_DIR | wx.FD_MULTIPLE) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None  # the user changed their mind
            else:
                # Proceed loading the file chosen by the user
                return fileDialog.GetPaths()

    def getDirDialog(self):
        with wx.DirDialog (None, "Choose input directory", "",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_CHANGE_DIR) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return None  # the user changed their mind
            else:
                # Proceed loading the dir chosen by the user
                return dirDialog.GetPath()
class Signal:
    def __init__(self, name, parent):
        self.name = name
        self.full_name = "{}/{}".format(parent.filename, self.name)
        self.selected = False
        self.color = random.choice(Dark2[8])
        self.line = None
        self.bands = None
        self.bokeh_source = parent.bokeh_source
        self.min_val = 0
        self.max_val = 0
        self.axis = 'default'
        self.sub_signals = []
        for name in self.bokeh_source.data.keys():
            if (len(name.split('/')) == 1 and name == self.name) or '/'.join(name.split('/')[:-1]) == self.name:
                self.sub_signals.append(name)
        if len(self.sub_signals) > 1:
            self.mean_signal = squeeze_list([name for name in self.sub_signals if 'Mean' in name.split('/')[-1]])
            self.stdev_signal = squeeze_list([name for name in self.sub_signals if 'Stdev' in name.split('/')[-1]])
            self.min_signal = squeeze_list([name for name in self.sub_signals if 'Min' in name.split('/')[-1]])
            self.max_signal = squeeze_list([name for name in self.sub_signals if 'Max' in name.split('/')[-1]])
        else:
            self.mean_signal = squeeze_list(self.name)
            self.stdev_signal = None
            self.min_signal = None
            self.max_signal = None
        self.has_bollinger_bands = False
        if self.mean_signal and self.stdev_signal and self.min_signal and self.max_signal:
            self.has_bollinger_bands = True
        self.show_bollinger_bands = False
        self.bollinger_bands_source = None
        self.update_range()

    def set_color(self, color):
        self.color = color
        if self.line:
            self.line.glyph.line_color = color
        if self.bands:
            self.bands.glyph.fill_color = color

    def set_selected(self, val):
        global current_color
        if self.selected != val:
            self.selected = val
            if self.line:
                # self.set_color(Dark2[8][current_color])
                # current_color = (current_color + 1) % len(Dark2[8])
                self.line.visible = self.selected
                if self.bands:
                    self.bands.visible = self.selected and self.show_bollinger_bands
            elif self.selected:
                # lazy plotting - plot only when selected for the first time
                show_spinner()
                self.set_color(Dark2[8][current_color])
                current_color = (current_color + 1) % len(Dark2[8])
                if self.has_bollinger_bands:
                    self.set_bands_source()
                    self.create_bands()
                self.line = plot.line('index', self.mean_signal, source=self.bokeh_source,
                                      line_color=self.color, line_width=2)
                self.line.visible = True
                hide_spinner()

    def set_dash(self, dash):
        self.line.glyph.line_dash = dash

    def create_bands(self):
        self.bands = plot.patch(x='band_x', y='band_y', source=self.bollinger_bands_source,
                                color=self.color, fill_alpha=0.4, alpha=0.1, line_width=0)
        self.bands.visible = self.show_bollinger_bands
        # self.min_line = plot.line('index', self.min_signal, source=self.bokeh_source,
        #                           line_color=self.color, line_width=3, line_dash="4 4")
        # self.max_line = plot.line('index', self.max_signal, source=self.bokeh_source,
        #                           line_color=self.color, line_width=3, line_dash="4 4")
        # self.min_line.visible = self.show_bollinger_bands
        # self.max_line.visible = self.show_bollinger_bands

    def set_bands_source(self):
        x_ticks = self.bokeh_source.data['index']
        mean_values = self.bokeh_source.data[self.mean_signal]
        stdev_values = self.bokeh_source.data[self.stdev_signal]
        band_x = np.append(x_ticks, x_ticks[::-1])
        band_y = np.append(mean_values - stdev_values, mean_values[::-1] + stdev_values[::-1])
        source_data = {'band_x': band_x, 'band_y': band_y}
        if self.bollinger_bands_source:
            self.bollinger_bands_source.data = source_data
        else:
            self.bollinger_bands_source = ColumnDataSource(source_data)

    def change_bollinger_bands_state(self, new_state):
        self.show_bollinger_bands = new_state
        if self.bands and self.selected:
            self.bands.visible = new_state
            # self.min_line.visible = new_state
            # self.max_line.visible = new_state

    def update_range(self):
        self.min_val = np.min(self.bokeh_source.data[self.mean_signal])
        self.max_val = np.max(self.bokeh_source.data[self.mean_signal])

    def set_axis(self, axis):
        self.axis = axis
        self.line.y_range_name = axis

    def toggle_axis(self):
        if self.axis == 'default':
            self.set_axis('secondary')
        else:
            self.set_axis('default')


class SignalsFileBase:
    def __init__(self):
        self.full_csv_path = ""
        self.dir = ""
        self.filename = ""
        self.signals_averaging_window = 1
        self.show_bollinger_bands = False
        self.csv = None
        self.bokeh_source = None
        self.bokeh_source_orig = None
        self.last_modified = None
        self.signals = {}
        self.separate_files = False

    def load_csv(self):
        pass

    def update_source_and_signals(self):
        # create bokeh data sources
        self.bokeh_source_orig = ColumnDataSource(self.csv)
        self.bokeh_source_orig.data['index'] = self.bokeh_source_orig.data[x_axis]

        if self.bokeh_source is None:
            self.bokeh_source = ColumnDataSource(self.csv)
        else:
            # self.bokeh_source.data = self.bokeh_source_orig.data
            # smooth the data if necessary
            self.change_averaging_window(self.signals_averaging_window, force=True)

        # create all the signals
        if len(self.signals.keys()) == 0:
            self.signals = {}
            unique_signal_names = []
            for name in self.csv.columns:
                if len(name.split('/')) == 1:
                    unique_signal_names.append(name)
                else:
                    unique_signal_names.append('/'.join(name.split('/')[:-1]))
            unique_signal_names = list(set(unique_signal_names))
            for signal_name in unique_signal_names:
                self.signals[signal_name] = Signal(signal_name, self)

    def load(self):
        self.load_csv()
        self.update_source_and_signals()

    def reload_data(self, signals):
        # this function is a workaround to reload the data of all the signals
        # if the data doesn't change, bokeh does not refreshes the line
        self.change_averaging_window(self.signals_averaging_window + 1, force=True)
        self.change_averaging_window(self.signals_averaging_window - 1, force=True)

    def change_averaging_window(self, new_size, force=False, signals=None):
        if force or self.signals_averaging_window != new_size:
            self.signals_averaging_window = new_size
            win = np.ones(new_size) / new_size
            temp_data = self.bokeh_source_orig.data.copy()
            for col in self.bokeh_source.data.keys():
                if col == 'index' or col in x_axis_options \
                        or (signals and not any(col in signal for signal in signals)):
                    temp_data[col] = temp_data[col][:-new_size]
                    continue
                temp_data[col] = np.convolve(self.bokeh_source_orig.data[col], win, mode='same')[:-new_size]
            self.bokeh_source.data = temp_data

            # smooth bollinger bands
            for signal in self.signals.values():
                if signal.has_bollinger_bands:
                    signal.set_bands_source()

    def hide_all_signals(self):
        for signal_name in self.signals.keys():
            self.set_signal_selection(signal_name, False)

    def set_signal_selection(self, signal_name, val):
        self.signals[signal_name].set_selected(val)

    def change_bollinger_bands_state(self, new_state):
        self.show_bollinger_bands = new_state
        for signal in self.signals.values():
            signal.change_bollinger_bands_state(new_state)

    def file_was_modified_on_disk(self):
        pass

    def get_range_of_selected_signals_on_axis(self, axis, selected_signal=None):
        max_val = -float('inf')
        min_val = float('inf')
        for signal in self.signals.values():
            if (selected_signal and signal.name == selected_signal) or (signal.selected and signal.axis == axis):
                max_val = max(max_val, signal.max_val)
                min_val = min(min_val, signal.min_val)
        return min_val, max_val

    def get_selected_signals(self):
        signals = []
        for signal in self.signals.values():
            if signal.selected:
                signals.append(signal)
        return signals

    def show_files_separately(self, val):
        pass


class SignalsFile(SignalsFileBase):
    def __init__(self, csv_path, load=True):
        SignalsFileBase.__init__(self)
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

        self.last_modified = os.path.getmtime(self.full_csv_path)

    def file_was_modified_on_disk(self):
        return self.last_modified != os.path.getmtime(self.full_csv_path)


class SignalsFilesGroup(SignalsFileBase):
    def __init__(self, csv_paths):
        SignalsFileBase.__init__(self)
        self.full_csv_paths = csv_paths
        self.signals_files = []
        if len(csv_paths) == 1 and os.path.isdir(csv_paths[0]):
            self.signals_files = [SignalsFile(str(file), load=False) for file in add_directory_csv_files(csv_paths[0])]
        else:
            for csv_path in csv_paths:
                if os.path.isdir(csv_path):
                    self.signals_files.append(SignalsFilesGroup(add_directory_csv_files(csv_path)))
                else:
                    self.signals_files.append(SignalsFile(str(csv_path), load=False))
        if len(csv_paths) == 1:
            # get the parent directory name (since the current directory is the timestamp directory)
            self.dir = os.path.abspath(os.path.join(os.path.dirname(csv_paths[0]), '..'))
        else:
            # get the common directory for all the experiments
            self.dir = os.path.dirname(os.path.commonprefix(csv_paths))
        self.filename = '{} - Group({})'.format(basename(self.dir), len(self.signals_files))
        self.load()

        # this helps set the correct x axis
        self.change_averaging_window(1, force=True)

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

        # remove NaNs
        self.csv.fillna(value=0, inplace=True)  # removing this line will make bollinger bands fail
        for key in self.csv.keys():
            if 'Stdev' in key and 'Evaluation' not in key:
                self.csv[key] = self.csv[key].fillna(value=0)

        for signal_file in self.signals_files:
            signal_file.update_source_and_signals()

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


class RunType(Enum):
    SINGLE_FOLDER_SINGLE_FILE = 1
    SINGLE_FOLDER_MULTIPLE_FILES = 2
    MULTIPLE_FOLDERS_SINGLE_FILES = 3
    MULTIPLE_FOLDERS_MULTIPLE_FILES = 4
    UNKNOWN = 0


class FolderType(Enum):
    SINGLE_FILE = 1
    MULTIPLE_FILES = 2
    MULTIPLE_FOLDERS = 3
    EMPTY = 4

dialog = DialogApp()

# read data
patches = {}
signals_files = {}
selected_file = None
x_axis = 'Episode #'
x_axis_options = ['Episode #', 'Total steps', 'Wall-Clock Time']
current_color = 0

# spinner
root_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(root_dir, 'spinner.css'), 'r') as f:
    spinner_style = """<style>{}</style>""".format(f.read())
spinner_html = """<ul class="spinner"><li></li><li></li><li></li><li></li></ul>"""
spinner = Div(text="""""")

# file refresh time placeholder
refresh_info = Div(text="""""", width=210)

# create figures
plot = figure(plot_width=1200, plot_height=800,
              tools='pan,box_zoom,wheel_zoom,crosshair,undo,redo,reset,save',
              toolbar_location='above', x_axis_label='Episodes',
              x_range=Range1d(0, 10000), y_range=Range1d(0, 100000))
plot.extra_y_ranges = {"secondary": Range1d(start=-100, end=200)}
plot.add_layout(LinearAxis(y_range_name="secondary"), 'right')

# legend
div = Div(text="""""")
legend = widgetbox([div])

bokeh_legend = Legend(
    items=[("12345678901234567890123456789012345678901234567890", [])],  # 50 letters
    # items=[("                                                  ", [])],  # 50 letters
    location=(-20, 0), orientation="vertical",
    border_line_color="black",
    label_text_font_size={'value': '9pt'},
    margin=30
)
plot.add_layout(bokeh_legend, "right")


def update_axis_range(name, range_placeholder):
    max_val = -float('inf')
    min_val = float('inf')
    selected_signal = None
    if name in x_axis_options:
        selected_signal = name
    for signals_file in signals_files.values():
        curr_min_val, curr_max_val = signals_file.get_range_of_selected_signals_on_axis(name, selected_signal)
        max_val = max(max_val, curr_max_val)
        min_val = min(min_val, curr_min_val)
    if min_val != float('inf'):
        range = max_val - min_val
        range_placeholder.start = min_val - 0.1 * range
        range_placeholder.end = max_val + 0.1 * range


# update axes ranges
def update_ranges():
    update_axis_range('default', plot.y_range)
    update_axis_range('secondary', plot.extra_y_ranges['secondary'])


def get_all_selected_signals():
    signals = []
    for signals_file in signals_files.values():
        signals += signals_file.get_selected_signals()
    return signals


# update legend using the legend text dictionary
def update_legend():
    legend_text = """<div></div>"""
    selected_signals = get_all_selected_signals()
    items = []
    for signal in selected_signals:
        side_sign = "<" if signal.axis == 'default' else ">"
        legend_text += """<div style='color: {}'><b>{} {}</b></div>"""\
                       .format(signal.color, side_sign, signal.full_name)
        items.append((signal.full_name, [signal.line]))
    div.text = legend_text
    # the visible=false => visible=true is a hack to make the legend render again
    bokeh_legend.visible = False
    bokeh_legend.items = items
    bokeh_legend.visible = True


# select lines to display
def select_data(args, old, new):
    if selected_file is None:
        return
    show_spinner()
    selected_signals = new
    for signal_name in selected_file.signals.keys():
        is_selected = signal_name in selected_signals
        selected_file.set_signal_selection(signal_name, is_selected)

    # update axes ranges
    update_ranges()
    update_axis_range(x_axis, plot.x_range)

    # update the legend
    update_legend()

    hide_spinner()


# add new lines to the plot
def plot_signals(signals_file, signals):
    for idx, signal in enumerate(signals):
        signal.line = plot.line('index', signal.name, source=signals_file.bokeh_source,
                                line_color=signal.color, line_width=2)


def open_file_dialog():
    return dialog.getFileDialog()


def open_directory_dialog():
    return dialog.getDirDialog()


def show_spinner():
    spinner.text = spinner_style + spinner_html


def hide_spinner():
    spinner.text = ""


# will create a group from the files
def create_files_group_signal(files):
    global selected_file
    signals_file = SignalsFilesGroup(files)
    signals_files[signals_file.filename] = signals_file

    filenames = [signals_file.filename]
    files_selector.options += filenames
    files_selector.value = filenames[0]
    selected_file = signals_file


# load files from disk as a group
def load_files_group():
    show_spinner()
    files = open_file_dialog()
    # no files selected
    if not files or not files[0]:
        hide_spinner()
        return

    change_displayed_doc()

    if len(files) == 1:
        create_files_signal(files)
    else:
        create_files_group_signal(files)

    change_selected_signals_in_data_selector([""])
    hide_spinner()


# classify the folder as containing a single file, multiple files or only folders
def classify_folder(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]
    folders = [d for d in listdir(dir_path) if isdir(join(dir_path, d))]
    if len(files) == 1:
        return FolderType.SINGLE_FILE
    elif len(files) > 1:
        return FolderType.MULTIPLE_FILES
    elif len(folders) >= 1:
        return FolderType.MULTIPLE_FOLDERS
    else:
        return FolderType.EMPTY


# finds if this is single-threaded or multi-threaded
def get_run_type(dir_path):
    folder_type = classify_folder(dir_path)
    if folder_type == FolderType.SINGLE_FILE:
        return RunType.SINGLE_FOLDER_SINGLE_FILE

    elif folder_type == FolderType.MULTIPLE_FILES:
        return RunType.SINGLE_FOLDER_MULTIPLE_FILES

    elif folder_type == FolderType.MULTIPLE_FOLDERS:
        # folder contains sub dirs -> we assume we can classify the folder using only the first sub dir
        sub_dirs = [d for d in listdir(dir_path) if isdir(join(dir_path, d))]

        # checking only the first folder in the root dir for its type, since we assume that all sub dirs will share the
        # same structure (i.e. if one is a result of multi-threaded run, so will all the other).
        folder_type = classify_folder(os.path.join(dir_path, sub_dirs[0]))
        if folder_type == FolderType.SINGLE_FILE:
            folder_type = RunType.MULTIPLE_FOLDERS_SINGLE_FILES
        elif folder_type == FolderType.MULTIPLE_FILES:
            folder_type = RunType.MULTIPLE_FOLDERS_MULTIPLE_FILES
    return folder_type


# takes path to dir and recursively adds all it's files to paths
def add_directory_csv_files(dir_path, paths=None):
    if not paths:
        paths = []

    for p in listdir(dir_path):
        path = join(dir_path, p)
        if isdir(path):
            # call recursively for each dir
            paths = add_directory_csv_files(path, paths)
        elif isfile(path) and path.endswith('.csv'):
            # add every file to the list
            paths.append(path)

    return paths


# create a signal file from the directory path according to the directory underlying structure
def handle_dir(dir_path, run_type):
    paths = add_directory_csv_files(dir_path)
    if run_type == RunType.SINGLE_FOLDER_SINGLE_FILE:
        create_files_signal(paths)
    elif run_type == RunType.SINGLE_FOLDER_MULTIPLE_FILES:
        create_files_group_signal(paths)
    elif run_type == RunType.MULTIPLE_FOLDERS_SINGLE_FILES:
        create_files_group_signal(paths)
    elif run_type == RunType.MULTIPLE_FOLDERS_MULTIPLE_FILES:
        sub_dirs = [d for d in listdir(dir_path) if isdir(join(dir_path, d))]
        # for d in sub_dirs:
        #     paths = add_directory_csv_files(os.path.join(dir_path, d))
        #     create_files_group_signal(paths)
        create_files_group_signal([os.path.join(dir_path, d) for d in sub_dirs])


# load directory from disk as a group
def load_directory_group():
    global selected_file
    show_spinner()
    directory = open_directory_dialog()
    # no files selected
    if not directory:
        hide_spinner()
        return

    change_displayed_doc()

    handle_dir(directory, get_run_type(directory))

    change_selected_signals_in_data_selector([""])
    hide_spinner()


def create_files_signal(files):
    global selected_file
    new_signal_files = []
    for idx, file_path in enumerate(files):
        signals_file = SignalsFile(str(file_path))
        signals_files[signals_file.filename] = signals_file
        new_signal_files.append(signals_file)

    filenames = [f.filename for f in new_signal_files]

    files_selector.options += filenames
    files_selector.value = filenames[0]
    selected_file = new_signal_files[0]


# load files from disk
def load_files():
    show_spinner()
    files = open_file_dialog()

    # no files selected
    if not files or not files[0]:
        hide_spinner()
        return

    create_files_signal(files)
    hide_spinner()

    change_selected_signals_in_data_selector([""])


def unload_file():
    global selected_file
    global signals_files
    if selected_file is None:
        return
    selected_file.hide_all_signals()
    del signals_files[selected_file.filename]
    data_selector.options = [""]
    filenames = cycle(files_selector.options)
    files_selector.options.remove(selected_file.filename)
    if len(files_selector.options) > 0:
        files_selector.value = next(filenames)
    else:
        files_selector.value = None
    update_legend()
    refresh_info.text = ""


# reload the selected csv file
def reload_all_files(force=False):
    for file_to_load in signals_files.values():
        if force or file_to_load.file_was_modified_on_disk():
            file_to_load.load()
        refresh_info.text = "last update: " + str(datetime.datetime.now()).split(".")[0]


# unselect the currently selected signals and then select the requested signals in the data selector
def change_selected_signals_in_data_selector(selected_signals):
    # the default bokeh way is not working due to a bug since Bokeh 0.12.6 (https://github.com/bokeh/bokeh/issues/6501)
    # this will currently cause the signals to change color
    for value in list(data_selector.value):
        if value in data_selector.options:
            index = data_selector.options.index(value)
            data_selector.options.remove(value)
            data_selector.value.remove(value)
            data_selector.options.insert(index, value)
    data_selector.value = selected_signals


# change data options according to the selected file
def change_data_selector(args, old, new):
    global selected_file
    if new is None:
        selected_file = None
        return
    show_spinner()
    selected_file = signals_files[new]
    data_selector.options = sorted(list(selected_file.signals.keys()))
    selected_signal_names = [s.name for s in selected_file.signals.values() if s.selected]
    if not selected_signal_names:
        selected_signal_names = [""]
    change_selected_signals_in_data_selector(selected_signal_names)
    averaging_slider.value = selected_file.signals_averaging_window
    group_cb.active = [0 if selected_file.show_bollinger_bands else None]
    group_cb.active += [1 if selected_file.separate_files else None]
    hide_spinner()


# smooth all the signals of the selected file
def update_averaging(args, old, new):
    show_spinner()
    selected_file.change_averaging_window(new)
    hide_spinner()


def change_x_axis(val):
    global x_axis
    show_spinner()
    x_axis = x_axis_options[val]
    plot.xaxis.axis_label = x_axis
    reload_all_files(force=True)
    update_axis_range(x_axis, plot.x_range)
    hide_spinner()


# move the signal between the main and secondary Y axes
def toggle_second_axis():
    show_spinner()
    signals = selected_file.get_selected_signals()
    for signal in signals:
        signal.toggle_axis()

    update_ranges()
    update_legend()

    # this is just for redrawing the signals
    selected_file.reload_data([signal.name for signal in signals])

    hide_spinner()


def toggle_group_property(new):
    # toggle show / hide Bollinger bands
    selected_file.change_bollinger_bands_state(0 in new)

    # show a separate signal for each file in a group
    selected_file.show_files_separately(1 in new)


def change_displayed_doc():
    if doc.roots[0] == landing_page:
        doc.remove_root(landing_page)
        doc.add_root(layout)


# Color selection - most of these functions are taken from bokeh examples (plotting/color_sliders.py)
def select_color(attr, old, new):
    show_spinner()
    signals = selected_file.get_selected_signals()
    for signal in signals:
        signal.set_color(rgb_to_hex(crRGBs[new['1d']['indices'][0]]))
    hide_spinner()


def generate_color_range(N, I):
    HSV_tuples = [(x*1.0/N, 0.5, I) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    for_conversion = []
    for RGB_tuple in RGB_tuples:
        for_conversion.append((int(RGB_tuple[0]*255), int(RGB_tuple[1]*255), int(RGB_tuple[2]*255)))
    hex_colors = [rgb_to_hex(RGB_tuple) for RGB_tuple in for_conversion]
    return hex_colors, for_conversion


# convert RGB tuple to hexadecimal code
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# convert hexadecimal to RGB tuple
def hex_to_dec(hex):
    red = ''.join(hex.strip('#')[0:2])
    green = ''.join(hex.strip('#')[2:4])
    blue = ''.join(hex.strip('#')[4:6])
    return int(red, 16), int(green, 16), int(blue,16)

color_resolution = 1000
brightness = 0.75  # change to have brighter/darker colors
crx = list(range(1, color_resolution+1))  # the resolution is 1000 colors
cry = [5 for i in range(len(crx))]
crcolor, crRGBs = generate_color_range(color_resolution, brightness)  # produce spectrum


# ---------------- Build Website Layout -------------------

# select file
file_selection_button = Button(label="Select Files", button_type="success", width=120)
file_selection_button.on_click(load_files_group)

files_selector_spacer = Spacer(width=10)

group_selection_button = Button(label="Select Directory", button_type="primary", width=140)
group_selection_button.on_click(load_directory_group)

unload_file_button = Button(label="Unload", button_type="danger", width=50)
unload_file_button.on_click(unload_file)

# files selection box
files_selector = Select(title="Files:", options=[], width=200)
files_selector.on_change('value', change_data_selector)

# data selection box
data_selector = MultiSelect(title="Data:", options=[], size=12)
data_selector.on_change('value', select_data)

# x axis selection box
x_axis_selector_title = Div(text="""X Axis:""")
x_axis_selector = RadioButtonGroup(labels=x_axis_options, active=0)
x_axis_selector.on_click(change_x_axis)

# toggle second axis button
toggle_second_axis_button = Button(label="Toggle Second Axis", button_type="success")
toggle_second_axis_button.on_click(toggle_second_axis)

# averaging slider
averaging_slider = Slider(title="Averaging window", start=1, end=101, step=10)
averaging_slider.on_change('value', update_averaging)

# group properties checkbox
group_cb = CheckboxGroup(labels=["Show statistics bands", "Ungroup signals"], active=[])
group_cb.on_click(toggle_group_property)

# color selector
color_selector_title = Div(text="""Select Color:""")
crsource = ColumnDataSource(data=dict(x=crx, y=cry, crcolor=crcolor, RGBs=crRGBs))
color_selector = figure(x_range=(0, color_resolution), y_range=(0, 10),
                        plot_width=300, plot_height=40,
                        tools='tap')
color_selector.axis.visible = False
color_range = color_selector.rect(x='x', y='y', width=1, height=10,
                                  color='crcolor', source=crsource)
crsource.on_change('selected', select_color)
color_range.nonselection_glyph = color_range.glyph
color_selector.toolbar.logo = None
color_selector.toolbar_location = None

# title
title = Div(text="""<h1>Coach Dashboard</h1>""")

# landing page
landing_page_description = Div(text="""<h3>Start by selecting an experiment file or directory to open:</h3>""")
center = Div(text="""<style>html { text-align: center; } </style>""")
center_buttons = Div(text="""<style>.bk-grid-row .bk-layout-fixed { margin: 0 auto; }</style>""", width=0)
landing_page = column(center,
                      title,
                      landing_page_description,
                      row(center_buttons),
                      row(file_selection_button, sizing_mode='scale_width'),
                      row(group_selection_button, sizing_mode='scale_width'),
                      sizing_mode='scale_width')

# main layout of the document
layout = row(file_selection_button, files_selector_spacer, group_selection_button, width=300)
layout = column(layout, files_selector)
layout = column(layout, row(refresh_info, unload_file_button))
layout = column(layout, data_selector)
layout = column(layout, color_selector_title)
layout = column(layout, color_selector)
layout = column(layout, x_axis_selector_title)
layout = column(layout, x_axis_selector)
layout = column(layout, group_cb)
layout = column(layout, toggle_second_axis_button)
layout = column(layout, averaging_slider)
# layout = column(layout, legend)
layout = row(layout, plot)
layout = column(title, layout)
layout = column(layout, spinner)

doc = curdoc()
doc.add_root(landing_page)

doc.add_periodic_callback(reload_all_files, 20000)
plot.y_range = Range1d(0, 100)
plot.extra_y_ranges['secondary'] = Range1d(0, 100)

# show load file dialog immediately on start
#doc.add_timeout_callback(load_files, 1000)

if __name__ == "__main__":
    # find an open port and run the server
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 12345
    while True:
        try:
            s.bind(("127.0.0.1", port))
            break
        except socket.error as e:
            if e.errno == 98:
                port += 1
    s.close()
    os.system('bokeh serve --show dashboard.py --port {}'.format(port))
