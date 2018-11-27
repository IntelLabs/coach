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


from bokeh.layouts import row, column, widgetbox, Spacer
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Legend
from bokeh.models.widgets import RadioButtonGroup, MultiSelect, Button, Select, Slider, Div, CheckboxGroup, Toggle
from bokeh.plotting import figure
from rl_coach.dashboard_components.globals import layouts, crcolor, crx, cry, color_resolution, crRGBs
from rl_coach.dashboard_components.experiment_board import file_selection_button, files_selector_spacer, \
    group_selection_button, unload_file_button, files_selector

# ---------------- Build Website Layout -------------------

# file refresh time placeholder
refresh_info = Div(text="""""", width=210)

# create figures
plot = figure(plot_width=1200, plot_height=800,
              tools='pan,box_zoom,wheel_zoom,crosshair,undo,redo,reset,save',
              toolbar_location='above', x_axis_label='Episodes',
              x_range=Range1d(0, 10000), y_range=Range1d(0, 100000))
plot.extra_y_ranges = {"secondary": Range1d(start=-100, end=200)}
plot.add_layout(LinearAxis(y_range_name="secondary"), 'right')
plot.yaxis[-1].visible = False

# legend
div = Div(text="""""")
legend = widgetbox([div])

bokeh_legend = Legend(
    # items=[("12345678901234567890123456789012345678901234567890", [])],  # 50 letters
    items=[("__________________________________________________", [])],  # 50 letters
    location=(0, 0), orientation="vertical",
    border_line_color="black",
    label_text_font_size={'value': '9pt'},
    margin=30
)
plot.add_layout(bokeh_legend, "right")

# select file
file_selection_button = Button(label="Select Files", button_type="success", width=120)
# file_selection_button.on_click(load_files_group)

files_selector_spacer = Spacer(width=10)

group_selection_button = Button(label="Select Directory", button_type="primary", width=140)
# group_selection_button.on_click(load_directory_group)

unload_file_button = Button(label="Unload", button_type="danger", width=50)
# unload_file_button.on_click(unload_file)

# files selection box
files_selector = Select(title="Files:", options=[])
# files_selector.on_change('value', change_data_selector)

# data selection box
data_selector = MultiSelect(title="Data:", options=[], size=12)
# data_selector.on_change('value', select_data)

# toggle second axis button
toggle_second_axis_button = Button(label="Toggle Second Axis", button_type="success")
# toggle_second_axis_button.on_click(toggle_second_axis)

# averaging slider
averaging_slider = Slider(title="Averaging window", start=1, end=101, step=10)
# averaging_slider.on_change('value', update_averaging)

# color selector
color_selector_title = Div(text="""Select Color:""")
crsource = ColumnDataSource(data=dict(x=crx, y=cry, crcolor=crcolor, RGBs=crRGBs))
color_selector = figure(x_range=(0, color_resolution), y_range=(0, 10),
                        plot_width=300, plot_height=40,
                        tools='tap')
color_selector.axis.visible = False
color_range = color_selector.rect(x='x', y='y', width=1, height=10,
                                  color='crcolor', source=crsource)
# crsource.on_change('selected', select_color)
color_range.nonselection_glyph = color_range.glyph
color_selector.toolbar.logo = None
color_selector.toolbar_location = None

episode_selector = MultiSelect(title="Episode:", options=['0', '1', '2', '3', '4'], size=1)

online_toggle = Toggle(label="Online", button_type="success")

# main layout of the document
layout = row(file_selection_button, files_selector_spacer, group_selection_button, width=300)
layout = column(layout, files_selector)
layout = column(layout, row(refresh_info, unload_file_button))
layout = column(layout, data_selector)
layout = column(layout, color_selector_title)
layout = column(layout, color_selector)
layout = column(layout, toggle_second_axis_button)
layout = column(layout, averaging_slider)
layout = column(layout, episode_selector)
layout = column(layout, online_toggle)
layout = row(layout, plot)

episodic_board_layout = layout

layouts["episodic_board"] = episodic_board_layout