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

from pandas import DataFrame
import os
import re
from pprint import pprint
import threading
from subprocess import Popen, PIPE
import time
import datetime
from six.moves import input
from PIL import Image
from typing import Union
import shutil

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

    def ask_yes_no(self, title: str, default: Union[None, bool]=None):
        """
        Ask the user for a yes / no question and return True if the answer is yes and False otherwise.
        The function will keep asking the user for an answer until he answers one of the possible responses.
        A default answer can be passed and will be selected if the user presses enter
        :param title: The question to ask the user
        :param default: the default answer
        :return: True / False according to the users answer
        """
        default_answer = 'y/n'
        if default == True:
            default_answer = 'Y/n'
        elif default == False:
            default_answer = 'y/N'

        while True:
            answer = input("{}{}{} ({})".format(Colors.BG_CYAN, title, Colors.END, default_answer))
            if answer == "yes" or answer == "YES" or answer == "y" or answer == "Y":
                return True
            elif answer == "no" or answer == "NO" or answer == "n" or answer == "N":
                return False
            elif answer == "":
                if default is not None:
                    return default

    def change_terminal_title(self, title: str):
        """
        Changes the title of the terminal window
        :param title: The new title
        :return: None
        """
        print("\x1b]2;{}\x07".format(title))


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
        self.time_started = datetime.datetime.now()
        self.start_time = None
        self.time = None
        self.experiments_path = ""
        self.last_line_idx_written_to_csv = 0
        self.experiment_name = ""

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
            value = self.get_signal_value(time, signal_name)
            if value != value:  # value is nan
                return False
        except:
            return False
        return True

    def get_signal_value(self, time, signal_name):
        return self.data.loc[time, signal_name]

    def dump_output_csv(self, append=True):
        self.data.index.name = "Episode #"
        if len(self.data.index) == 1:
            self.start_time = time.time()

        if os.path.exists(self.csv_path) and append:
            self.data[self.last_line_idx_written_to_csv:].to_csv(self.csv_path, mode='a', header=False)
        else:
            self.data.to_csv(self.csv_path)

        self.last_line_idx_written_to_csv = len(self.data.index)

    def update_wall_clock_time(self, episode):
        if self.start_time:
            self.create_signal_value('Wall-Clock Time', time.time() - self.start_time, time=episode)
        else:
            self.create_signal_value('Wall-Clock Time', 0, time=episode)
            self.start_time = time.time()

    def create_gif(self, images, fps=10, name="Gif"):
        output_file = '{}_{}.gif'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name)
        output_dir = os.path.join(self.experiments_path, 'gifs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_file)
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=1.0 / fps, loop=0)

    def remove_experiment_dir(self):
        shutil.rmtree(self.experiments_path)

    def summarize_experiment(self):
        screen.separator()
        screen.log_title("Results stored at: {}".format(self.experiments_path))
        screen.log_title("Total runtime: {}".format(datetime.datetime.now() - self.time_started))
        if 'Training Reward' in self.data.keys() and 'Evaluation Reward' in self.data.keys():
            screen.log_title("Max training reward: {}, max evaluation reward: {}".format(self.data['Training Reward'].max(), self.data['Evaluation Reward'].max()))
        screen.separator()
        if screen.ask_yes_no("Do you want to discard the experiment results (Warning: this cannot be undone)?", False):
            self.remove_experiment_dir()
        elif screen.ask_yes_no("Do you want to specify a different experiment name to save to?", False):
            new_name = self.get_experiment_name()
            new_path = self.get_experiment_path(new_name, create_path=False)
            shutil.move(self.experiments_path, new_path)
            screen.log_title("Results moved to: {}".format(new_path))

    def get_experiment_name(self, initial_experiment_name=''):
        match = None
        while match is None:
            if initial_experiment_name == '':
                experiment_name = screen.ask_input("Please enter an experiment name: ")
            else:
                experiment_name = initial_experiment_name

            experiment_name = experiment_name.replace(" ", "_")
            match = re.match("^$|^[\w -/]{1,100}$", experiment_name)

            if match is None:
                screen.error('Experiment name must be composed only of alphanumeric letters, '
                             'underscores and dashes and should not be longer than 100 characters.')

        self.experiment_name = match.group(0)
        return self.experiment_name

    def get_experiment_path(self, experiment_name, create_path=True):
        general_experiments_path = os.path.join('./experiments/', experiment_name)

        cur_date = self.time_started.date()
        cur_time = self.time_started.time()

        if not os.path.exists(general_experiments_path) and create_path:
            os.makedirs(general_experiments_path)
        experiment_path = os.path.join(general_experiments_path, '{}_{}_{}-{}_{}'
                                       .format(logger.two_digits(cur_date.day), logger.two_digits(cur_date.month),
                                               cur_date.year, logger.two_digits(cur_time.hour),
                                               logger.two_digits(cur_time.minute)))
        i = 0
        while True:
            if os.path.exists(experiment_path):
                experiment_path = os.path.join(general_experiments_path, '{}_{}_{}-{}_{}_{}'
                                               .format(cur_date.day, cur_date.month, cur_date.year, cur_time.hour,
                                                       cur_time.minute, i))
                i += 1
            else:
                if create_path:
                    os.makedirs(experiment_path)
                return experiment_path


global logger
logger = Logger()

global screen
screen = ScreenLogger("")
