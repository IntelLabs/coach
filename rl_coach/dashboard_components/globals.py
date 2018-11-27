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
from genericpath import isdir, isfile
from os import listdir
from os.path import join
from enum import Enum
from bokeh.models import Div
from bokeh.plotting import curdoc
import wx
import colorsys

patches = {}
signals_files = {}
selected_file = None
x_axis = ['Episode #']
x_axis_options = ['Episode #', 'Total steps', 'Wall-Clock Time']
x_axis_labels = ['Episode #', 'Total steps (per worker)', 'Wall-Clock Time (minutes)']
current_color = 0

# spinner
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root_dir, 'dashboard_components/spinner.css'), 'r') as f:
    spinner_style = """<style>{}</style>""".format(f.read())
    spinner_html = """<ul class="spinner"><li></li><li></li><li></li><li></li>
                      <li>
                        <br>
                        <span style="font-size: 24px; font-weight: bold; margin-left: -175px; width: 400px; 
                        position: absolute; text-align: center;">
                            {}
                        </span>
                      </li></ul>"""
spinner = Div(text="""""")
displayed_doc = "landing_page"
layouts = {}


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


def display_boards():
    global displayed_doc
    if displayed_doc == "landing_page":
        doc.remove_root(doc.roots[0])
        doc.add_root(layouts["boards"])
        displayed_doc = "boards"


def show_spinner(text="Loading..."):
    spinner.text = spinner_style + spinner_html.format(text)


def hide_spinner():
    spinner.text = ""


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
        with wx.DirDialog(None, "Choose input directory", "",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_CHANGE_DIR) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return None  # the user changed their mind
            else:
                # Proceed loading the dir chosen by the user
                return dirDialog.GetPath()


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

doc = curdoc()
