import numpy as np
from bokeh.models import ColumnDataSource

from rl_coach.dashboard_components.signals import Signal
from rl_coach.dashboard_components.globals import x_axis, x_axis_options, show_spinner


class SignalsFileBase:
    def __init__(self, plot):
        self.plot = plot
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
        self.last_reload_data_fix = False

    def load_csv(self):
        pass

    def update_x_axis_index(self):
        global x_axis
        self.bokeh_source_orig.data['index'] = self.bokeh_source_orig.data[x_axis[0]]
        self.bokeh_source.data['index'] = self.bokeh_source.data[x_axis[0]]

    def toggle_y_axis(self, signal_name=None):
        if signal_name and signal_name in self.signals.keys():
            self.signals[signal_name].toggle_axis()
        else:
            for signal in self.signals.values():
                if signal.selected:
                    signal.toggle_axis()

    def update_source_and_signals(self):
        # create bokeh data sources
        self.bokeh_source_orig = ColumnDataSource(self.csv)

        if self.bokeh_source is None:
            self.bokeh_source = ColumnDataSource(self.csv)
            self.update_x_axis_index()
        else:
            self.update_x_axis_index()
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
                self.signals[signal_name] = Signal(signal_name, self, self.plot)

    def load(self):
        self.load_csv()
        self.update_source_and_signals()

    def reload_data(self):
        # this function is a workaround to reload the data of all the signals
        # if the data doesn't change, bokeh does not refresh the line
        temp_data = self.bokeh_source.data.copy()
        for col in self.bokeh_source.data.keys():
            if not self.last_reload_data_fix:
                temp_data[col] = temp_data[col][:-1]
        self.last_reload_data_fix = not self.last_reload_data_fix
        self.bokeh_source.data = temp_data

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