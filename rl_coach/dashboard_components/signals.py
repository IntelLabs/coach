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


import random

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2
from rl_coach.dashboard_components.globals import show_spinner, hide_spinner, current_color
from rl_coach.utils import squeeze_list


class Signal:
    def __init__(self, name, parent, plot):
        self.name = name
        self.full_name = "{}/{}".format(parent.filename, self.name)
        self.plot = plot
        self.selected = False
        self.color = random.choice(Dark2[8])
        self.line = None
        self.scatter = None
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

    def plot_line(self):
        global current_color
        self.set_color(Dark2[8][current_color])
        current_color = (current_color + 1) % len(Dark2[8])
        if self.has_bollinger_bands:
            self.set_bands_source()
            self.create_bands()
        self.line = self.plot.line('index', self.mean_signal, source=self.bokeh_source,
                                   line_color=self.color, line_width=2)
        # self.scatter = self.plot.scatter('index', self.mean_signal, source=self.bokeh_source)
        self.line.visible = True

    def set_selected(self, val):
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
                self.plot_line()

    def set_dash(self, dash):
        self.line.glyph.line_dash = dash

    def create_bands(self):
        self.bands = self.plot.patch(x='band_x', y='band_y', source=self.bollinger_bands_source,
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
        if not self.line:
            self.plot_line()
            self.line.visible = False
        self.line.y_range_name = axis

    def toggle_axis(self):
        if self.axis == 'default':
            self.set_axis('secondary')
        else:
            self.set_axis('default')
