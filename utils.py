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

import json
import inspect
import os
import numpy as np
import threading
from subprocess import call, Popen

killed_processes = []


class Enum:
    def __init__(self):
        pass

    def keys(self):
        return [attr.lower() for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def vals(self):
        vars = dict(inspect.getmembers(self, lambda a: not (inspect.isroutine(a))))
        return {key.lower(): vars[key] for key in vars}

    def get(self, string):
        if string.lower() in self.keys():
            return self.vals()[string.lower()]
        raise NameError('enum does not exist')

    def verify(self, string):
        if string.lower() in self.keys():
            return string.lower(), self.vals()[string.lower()]
        raise NameError('enum does not exist')

    def to_string(self, enum):
        for key, val in self.vals().items():
            if val == enum:
                return key
        raise NameError('enum does not exist')


class RunPhase(Enum):
    HEATUP = 0
    TRAIN = 1
    TEST = 2


def list_all_classes_in_module(module):
    return [k for k, v in inspect.getmembers(module, inspect.isclass) if v.__module__ == module.__name__]


def parse_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    return value


def convert_to_ascii(data):
    import collections
    if isinstance(data, basestring):
        return parse_bool(str(data))
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_to_ascii, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_to_ascii, data))
    else:
        return data


def break_file_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    extension = os.path.splitext(os.path.basename(path))[1]
    dir = os.path.dirname(path)
    return dir, base, extension


def is_empty(str):
    if str == 0:
        return True
    str = str.replace("'", "")
    str = str.replace("\"", "")
    if len(str) == 0:
        return True
    return False


def read_json(filename):
    # read json file
    with open(filename, 'r') as f:
        dict = json.loads(f.read())
        return dict


def write_json(filename, dict):
    # read json file
    with open(filename, 'w') as f:
        f.write(json.dumps(dict, indent=4))


def path_is_valid_dir(path):
    return os.path.isdir(path)


def remove_suffix(name, suffix_start):
    for s in suffix_start:
        split = name.find(s)
        if split != -1:
            name = name[:split]
            return name


def parse_int(value):
    import ast
    try:
        int_value = int(value)
        if int_value != value:
            return value
        else:
            return int_value
    except:
        pass

    try:
        return ast.literal_eval(value)
    except:
        return value


def set_gpu(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def set_cpu():
    set_gpu("")


# dictionary to class
class DictToClass(object):
    def __init__(self, data):
        for name, value in data.iteritems():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return DictToClass(value) if isinstance(value, dict) else value


# class to dictionary
def ClassToDict(x):
    # return dict((key, getattr(x, key)) for key in dir(x) if key not in dir(x.__class__))
    dictionary = x.__dict__
    return {key: dictionary[key] for key in dictionary.keys() if not key.startswith('__')}


def cmd_line_run(result, run_cmd, id=-1):
    p = Popen(run_cmd, shell=True, executable="/bin/bash")
    while result[0] is None or result[0] == [None]:
        if id in killed_processes:
            p.kill()
        result[0] = p.poll()


def threaded_cmd_line_run(run_cmd, id=-1):
    runThread = []
    result = [[None]]
    try:
        params = (result, run_cmd, id)
        runThread = threading.Thread(name='runThread', target=cmd_line_run, args=params)
        runThread.daemon = True
        runThread.start()
    except:
        runThread.join()
    return result


class Signal:
    def __init__(self, name):
        self.name = name
        self.sample_count = 0
        self.values = []

    def reset(self):
        self.sample_count = 0
        self.values = []

    def add_sample(self, sample):
        self.values.append(sample)

    def get_mean(self):
        if len(self.values) == 0:
            return ''
        if type(self.values[0]) == np.ndarray:
            return np.mean(np.concatenate(self.values))
        else:
            return np.mean(self.values)

    def get_max(self):
        if len(self.values) == 0:
            return ''
        if type(self.values[0]) == np.ndarray:
            return np.max(np.concatenate(self.values))
        else:
            return np.max(self.values)

    def get_min(self):
        if len(self.values) == 0:
            return ''
        if type(self.values[0]) == np.ndarray:
            return np.min(np.concatenate(self.values))
        else:
            return np.min(self.values)

    def get_stdev(self):
        if len(self.values) == 0:
            return ''
        if type(self.values[0]) == np.ndarray:
            return np.std(np.concatenate(self.values))
        else:
            return np.std(self.values)


def force_list(var):
    if isinstance(var, list):
        return var
    else:
        return [var]


def squeeze_list(var):
    if len(var) == 1:
        return var[0]
    else:
        return var


# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._shape = shape
        self._num_samples = 0
        self._mean = np.zeros(shape)
        self._std = np.zeros(shape)

    def reset(self):
        self._num_samples = 0
        self._mean = np.zeros(self._shape)
        self._std = np.zeros(self._shape)

    def push(self, sample):
        sample = np.asarray(sample)
        assert sample.shape == self._mean.shape, 'RunningStat input shape mismatch'
        self._num_samples += 1
        if self._num_samples == 1:
            self._mean[...] = sample
        else:
            old_mean = self._mean.copy()
            self._mean[...] = old_mean + (sample - old_mean) / self._num_samples
            self._std[...] = self._std + (sample - old_mean) * (sample - self._mean)

    @property
    def n(self):
        return self._num_samples

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._std / (self._num_samples - 1) if self._num_samples > 1 else np.square(self._mean)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._mean.shape


def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

