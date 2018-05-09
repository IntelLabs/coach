
import datetime
import os
import sys
import time
from itertools import cycle
from os import listdir
from os.path import isfile, join, isdir

from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column, Spacer
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Legend
from bokeh.models.widgets import RadioButtonGroup, MultiSelect, Button, Select, Slider, Div, CheckboxGroup
from bokeh.plotting import figure

from dashboard_components.globals import signals_files, x_axis_labels, x_axis_options, show_spinner, hide_spinner, \
    x_axis, dialog, FolderType, RunType, add_directory_csv_files, doc, display_boards, layouts, \
    crcolor, crx, cry, color_resolution, crRGBs, rgb_to_hex
from dashboard_components.signals_files_group import SignalsFilesGroup
from dashboard_components.signals_file import SignalsFile


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
def update_y_axis_ranges():
    update_axis_range('default', plot.y_range)
    update_axis_range('secondary', plot.extra_y_ranges['secondary'])


def update_x_axis_ranges():
    update_axis_range(x_axis[0], plot.x_range)


def get_all_selected_signals():
    signals = []
    for signals_file in signals_files.values():
        signals += signals_file.get_selected_signals()
    return signals


# update legend using the legend text dictionary
def update_legend():
    selected_signals = get_all_selected_signals()
    items = []
    for signal in selected_signals:
        side_sign = "◀" if signal.axis == 'default' else "▶"
        items.append((side_sign + " " + signal.full_name, [signal.line]))
    # the visible=false => visible=true is a hack to make the legend render again
    bokeh_legend.visible = False
    bokeh_legend.items = items  # this step takes a long time because it is redrawing the plot
    bokeh_legend.visible = True


# select lines to display
def select_data(args, old, new):
    if selected_file is None:
        return
    show_spinner("Updating the signal selection...")
    selected_signals = new
    for signal_name in selected_file.signals.keys():
        is_selected = signal_name in selected_signals
        selected_file.set_signal_selection(signal_name, is_selected)

    # update axes ranges
    update_y_axis_ranges()
    update_x_axis_ranges()

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


# will create a group from the files
def create_files_group_signal(files):
    global selected_file
    signals_file = SignalsFilesGroup(files, plot)
    signals_files[signals_file.filename] = signals_file

    filenames = [signals_file.filename]
    files_selector.options += filenames
    files_selector.value = filenames[0]
    selected_file = signals_file


# load files from disk as a group
def load_files_group():
    show_spinner("Loading files group...")
    files = open_file_dialog()
    # no files selected
    if not files or not files[0]:
        hide_spinner()
        return

    display_boards()

    if len(files) == 1:
        create_files_signal(files)
    else:
        create_files_group_signal(files)

    change_selected_signals_in_data_selector([""])
    hide_spinner()


# classify the folder as containing a single file, multiple files or only folders
def classify_folder(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]
    folders = [d for d in listdir(dir_path) if isdir(join(dir_path, d)) and any(f.endswith(".csv") for f in os.listdir(join(dir_path, d)))]
    if len(files) == 1:
        return FolderType.SINGLE_FILE
    elif len(files) > 1:
        return FolderType.MULTIPLE_FILES
    elif len(folders) == 1:
        return classify_folder(join(dir_path, folders[0]))
    elif len(folders) > 1:
        return FolderType.MULTIPLE_FOLDERS
    else:
        return FolderType.EMPTY


# finds if this is single-threaded or multi-threaded
def get_run_type(dir_path):
    folder_type = classify_folder(dir_path)
    if folder_type == FolderType.SINGLE_FILE:
        folder_type = RunType.SINGLE_FOLDER_SINGLE_FILE

    elif folder_type == FolderType.MULTIPLE_FILES:
        folder_type = RunType.SINGLE_FOLDER_MULTIPLE_FILES

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


# create a signal file from the directory path according to the directory underlying structure
def handle_dir(dir_path, run_type):
    paths = add_directory_csv_files(dir_path)
    if run_type in [RunType.SINGLE_FOLDER_SINGLE_FILE,
                    RunType.SINGLE_FOLDER_MULTIPLE_FILES,
                    RunType.MULTIPLE_FOLDERS_SINGLE_FILES]:
        create_files_group_signal(paths)
    elif run_type == RunType.MULTIPLE_FOLDERS_MULTIPLE_FILES:
        sub_dirs = [d for d in listdir(dir_path) if isdir(join(dir_path, d))]
        # for d in sub_dirs:
        #     paths = add_directory_csv_files(os.path.join(dir_path, d))
        #     create_files_group_signal(paths)
        create_files_group_signal([os.path.join(dir_path, d) for d in sub_dirs])


# load directory from disk as a group
def load_directory_group():
    show_spinner("Loading directories group...")
    directory = open_directory_dialog()
    # no files selected
    if not directory:
        hide_spinner()
        return

    display_directory_group(directory)


def display_directory_group(directory):
    display_boards()
    show_spinner("Loading directories group...")

    while get_run_type(directory) == FolderType.EMPTY:
        show_spinner("Waiting for experiment directory to get populated...")
        sys.stdout.write("Waiting for experiment directory to get populated...\r")
        time.sleep(10)

    handle_dir(directory, get_run_type(directory))

    change_selected_signals_in_data_selector([""])
    hide_spinner()


def create_files_signal(files):
    global selected_file
    new_signal_files = []
    for idx, file_path in enumerate(files):
        signals_file = SignalsFile(str(file_path), plot=plot)
        signals_files[signals_file.filename] = signals_file
        new_signal_files.append(signals_file)

    filenames = [f.filename for f in new_signal_files]

    files_selector.options += filenames
    files_selector.value = filenames[0]
    selected_file = new_signal_files[0]


# load files from disk
def load_files():
    show_spinner("Loading files...")
    files = open_file_dialog()

    # no files selected
    if not files or not files[0]:
        hide_spinner()
        return

    display_files(files)


def display_files(files):
    display_boards()
    show_spinner("Loading files...")

    create_files_signal(files)

    change_selected_signals_in_data_selector([""])
    hide_spinner()


def unload_file():
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
    # remove the data selection callback before updating the selector
    data_selector.remove_on_change('value', select_data)
    for value in list(data_selector.value):
        if value in data_selector.options:
            index = data_selector.options.index(value)
            data_selector.options.remove(value)
            data_selector.value.remove(value)
            data_selector.options.insert(index, value)
    data_selector.value = selected_signals
    # add back the data selection callback
    data_selector.on_change('value', select_data)


# change data options according to the selected file
def change_data_selector(args, old, new):
    global selected_file
    if new is None:
        selected_file = None
        return
    show_spinner("Updating selection...")
    selected_file = signals_files[new]
    data_selector.remove_on_change('value', select_data)
    data_selector.options = sorted(list(selected_file.signals.keys()))
    data_selector.on_change('value', select_data)
    selected_signal_names = [s.name for s in selected_file.signals.values() if s.selected]
    if not selected_signal_names:
        selected_signal_names = [""]
    change_selected_signals_in_data_selector(selected_signal_names)
    averaging_slider.value = selected_file.signals_averaging_window
    if len(averaging_slider_dummy_source.data['value']) > 0:
        averaging_slider_dummy_source.data['value'][0] = selected_file.signals_averaging_window
    group_cb.active = [0 if selected_file.show_bollinger_bands else None]
    group_cb.active += [1 if selected_file.separate_files else None]
    hide_spinner()


# smooth all the signals of the selected file
def update_averaging(args, old, new):
    show_spinner("Smoothing the signals...")
    # get the actual value from the dummy source
    new = averaging_slider_dummy_source.data['value'][0]
    selected_file.change_averaging_window(new)
    hide_spinner()


def change_x_axis(val):
    global x_axis
    show_spinner("Updating the X axis...")
    x_axis[0] = x_axis_options[val]
    plot.xaxis.axis_label = x_axis_labels[val]

    for file_to_load in signals_files.values():
        file_to_load.update_x_axis_index()

    update_axis_range(x_axis[0], plot.x_range)
    hide_spinner()


# move the signal between the main and secondary Y axes
def toggle_second_axis():
    show_spinner("Switching the Y axis...")
    plot.yaxis[-1].visible = True
    selected_file.toggle_y_axis()

    # this is just for redrawing the signals
    selected_file.reload_data()

    update_y_axis_ranges()
    update_legend()

    hide_spinner()


def toggle_group_property(new):
    # toggle show / hide Bollinger bands
    selected_file.change_bollinger_bands_state(0 in new)

    # show a separate signal for each file in a group
    selected_file.show_files_separately(1 in new)


# Color selection - most of these functions are taken from bokeh examples (plotting/color_sliders.py)
def select_color(attr, old, new):
    show_spinner("Changing signal color...")
    signals = selected_file.get_selected_signals()
    for signal in signals:
        signal.set_color(rgb_to_hex(crRGBs[new['1d']['indices'][0]]))
    hide_spinner()


doc.add_periodic_callback(reload_all_files, 20000)

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

bokeh_legend = Legend(
    # items=[("12345678901234567890123456789012345678901234567890", [])],  # 50 letters
    items=[("__________________________________________________", [])],  # 50 letters
    location=(0, 0), orientation="vertical",
    border_line_color="black",
    label_text_font_size={'value': '9pt'},
    margin=30
)
plot.add_layout(bokeh_legend, "right")
plot.y_range = Range1d(0, 100)
plot.extra_y_ranges['secondary'] = Range1d(0, 100)

# select file
file_selection_button = Button(label="Select Files", button_type="success", width=120)
file_selection_button.on_click(load_files_group)

files_selector_spacer = Spacer(width=10)

group_selection_button = Button(label="Select Directory", button_type="primary", width=140)
group_selection_button.on_click(load_directory_group)

unload_file_button = Button(label="Unload", button_type="danger", width=50)
unload_file_button.on_click(unload_file)

# files selection box
files_selector = Select(title="Files:", options=[])
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
# This data source is just used to communicate / trigger the real callback
averaging_slider_dummy_source = ColumnDataSource(data=dict(value=[]))
averaging_slider_dummy_source.on_change('data', update_averaging)
averaging_slider = Slider(title="Averaging window", start=1, end=101, step=10, callback_policy='mouseup')
averaging_slider.callback = CustomJS(args=dict(source=averaging_slider_dummy_source), code="""
    source.data = { value: [cb_obj.value] }
""")

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
layout = row(layout, plot)

experiment_board_layout = layout

layouts["experiment_board"] = experiment_board_layout
