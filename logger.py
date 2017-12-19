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

from pandas import *
import os
from pprint import pprint
import threading
from subprocess import Popen, PIPE
import time
import datetime
from six.moves import input
from PIL import Image

global failed_imports
failed_imports = []


class Colors(object):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[37m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_PURPLE = '\033[45m'
    BG_CYAN = '\033[30;46m'
    BG_WHITE = '\x1b[30;47m'
    BG_RESET = '\033[49m'
    BOLD = '\033[1m'
    UNDERLINE_ON = '\033[4m'
    UNDERLINE_OFF = '\033[24m'
    END = '\033[0m'


# prints to screen with a prefix identifying the origin of the print
class ScreenLogger(object):
    def __init__(self, name):
        self.name = name

    def separator(self):
        print("")
        print("--------------------------------")
        print("")

    def log(self, data):
        print(data)

    def log_dict(self, dict, prefix=""):
        str = "{}{}{} - ".format(Colors.PURPLE, prefix, Colors.END)
        for k, v in dict.items():
            str += "{}{}: {}{} ".format(Colors.BLUE, k, Colors.END, v)
        print(str)

    def log_title(self, title):
        print("{}{}{}".format(Colors.BG_CYAN, title, Colors.END))

    def success(self, text):
        print("{}{}{}".format(Colors.GREEN, text, Colors.END))

    def warning(self, text):
        print("{}{}{}".format(Colors.YELLOW, text, Colors.END))

    def error(self, text, crash=True):
        print("{}{}{}".format(Colors.RED, text, Colors.END))
        if crash:
            exit(1)

    def ask_input(self, title):
        return input("{}{}{}".format(Colors.BG_CYAN, title, Colors.END))


class BaseLogger(object):
    def __init__(self):
        pass

    def set_current_time(self, time):
        pass

    def set_dump_dir(self, path, task_id):
        pass

    def create_signal_value(self, signal_name, value, overwrite=True, time=None):
        pass

    def change_signal_value(self, signal_name, time, value):
        pass

    def signal_value_exists(self, time, signal_name):
        pass

    def get_signal_value(self, time, signal_name):
        pass

    def dump_output_csv(self):
        pass

    def update_wall_clock_time(self, episode):
        pass


class Logger(BaseLogger):
    def __init__(self):
        BaseLogger.__init__(self)
        self.data = DataFrame()
        self.csv_path = ''
        self.doc_path = ''
        self.aggregated_data_across_threads = None
        self.start_time = None
        self.time = None
        self.experiments_path = ""

    def set_current_time(self, time):
        self.time = time

    def two_digits(self, num):
        return '%02d' % num

    def set_dump_dir(self, experiments_path, task_id=None, add_timestamp=False, filename='worker'):
        self.experiments_path = experiments_path

        # set file names
        if task_id is not None:
            filename += "_{}".format(task_id)

        # add timestamp
        self.time_started = datetime.datetime.now()
        if add_timestamp:
            t = self.time_started.time()
            d = self.time_started.date()
            filename += '_{}_{}_{}-{}_{}'.format(self.two_digits(d.day), self.two_digits(d.month),
                                                 d.year, self.two_digits(t.hour), self.two_digits(t.minute))

        # add an index to the file in case there is already an experiment running with the same timestamp
        path_exists = True
        idx = 0
        while path_exists:
            self.csv_path = os.path.join(experiments_path, '{}_{}.csv'.format(filename, idx))
            self.doc_path = os.path.join(experiments_path, '{}_{}.doc'.format(filename, idx))
            path_exists = os.path.exists(self.csv_path) or os.path.exists(self.doc_path)
            idx += 1

    def create_signal_value(self, signal_name, value, overwrite=True, time=None):
        if not time:
            time = self.time
        # create only if it doesn't already exist
        if overwrite or not self.signal_value_exists(time, signal_name):
            self.data.loc[time, signal_name] = value
            return True
        return False

    def change_signal_value(self, signal_name, time, value):
        # change only if it already exists
        if self.signal_value_exists(time, signal_name):
            self.data.loc[time, signal_name] = value
            return True
        return False

    def signal_value_exists(self, time, signal_name):
        try:
            self.get_signal_value(time, signal_name)
        except:
            return False
        return True

    def get_signal_value(self, time, signal_name):
        return self.data.loc[time, signal_name]

    def dump_output_csv(self):
        self.data.index.name = "Episode #"
        if len(self.data.index) == 1:
            self.start_time = time.time()

        self.data.to_csv(self.csv_path)

    def update_wall_clock_time(self, episode):
        if self.start_time:
            self.create_signal_value('Wall-Clock Time', time.time() - self.start_time, time=episode)

    def create_gif(self, images, fps=10, name="Gif"):
        output_file = '{}_{}.gif'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name)
        output_dir = os.path.join(self.experiments_path, 'gifs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_file)
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=1.0 / fps, loop=0)


global logger
logger = Logger()

global screen
screen = ScreenLogger("")
