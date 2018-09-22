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

import importlib
import importlib.util
import inspect
import json
import os
import signal
import sys
import threading
import time
from multiprocessing import Manager
from subprocess import Popen
from typing import List, Tuple

import atexit
import numpy as np

killed_processes = []

eps = np.finfo(np.float32).eps


def lower_under_to_upper(s):
    s = s.replace('_', ' ')
    s = s.title()
    return s.replace(' ', '')


def get_base_dir():
    return os.path.dirname(os.path.realpath(__file__))


def list_all_presets():
    presets_path = os.path.join(get_base_dir(), 'presets')
    return [f.split('.')[0] for f in os.listdir(presets_path) if f.endswith('.py') and f != '__init__.py']


def list_all_classes_in_module(module):
    return [k for k, v in inspect.getmembers(module, inspect.isclass) if v.__module__ == module.__name__]


def parse_bool(value):
    return {'true': True, 'false': False}.get(value.strip().lower(), value)


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
    return str == 0 or len(str.replace("'", "").replace("\"", "")) == 0


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
        return int_value if int_value == value else value
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


class Signal(object):
    """
    Stores a stream of values and provides methods like get_mean and get_max
    which returns the statistics about accumulated values.
    """
    def __init__(self, name):
        self.name = name
        self.sample_count = 0
        self.values = []

    def reset(self):
        self.sample_count = 0
        self.values = []

    def add_sample(self, sample):
        """
        :param sample: either a single value or an array of values
        """
        self.values.append(sample)

    def _get_values(self):
        if type(self.values[0]) == np.ndarray:
            return np.concatenate(self.values)
        else:
            return self.values

    def get_last_value(self):
        if len(self.values) == 0:
            return np.nan
        else:
            return self._get_values()[-1]

    def get_mean(self):
        if len(self.values) == 0:
            return ''
        return np.mean(self._get_values())

    def get_max(self):
        if len(self.values) == 0:
            return ''
        return np.max(self._get_values())

    def get_min(self):
        if len(self.values) == 0:
            return ''
        return np.min(self._get_values())

    def get_stdev(self):
        if len(self.values) == 0:
            return ''
        return np.std(self._get_values())


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


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def switch_axes_order(observation, from_type='channels_first', to_type='channels_last'):
    """
    transpose an observation axes from channels_first to channels_last or vice versa
    :param observation: a numpy array
    :param from_type: can be 'channels_first' or 'channels_last'
    :param to_type: can be 'channels_first' or 'channels_last'
    :return: a new observation with the requested axes order
    """
    if from_type == to_type or len(observation.shape) == 1:
        return observation
    assert 2 <= len(observation.shape) <= 3, 'num axes of an observation must be 2 for a vector or 3 for an image'
    assert type(observation) == np.ndarray, 'observation must be a numpy array'
    if len(observation.shape) == 3:
        if from_type == 'channels_first' and to_type == 'channels_last':
            return np.transpose(observation, (1, 2, 0))
        elif from_type == 'channels_last' and to_type == 'channels_first':
            return np.transpose(observation, (2, 0, 1))
    else:
        return np.transpose(observation, (1, 0))


def stack_observation(curr_stack, observation, stack_size):
    """
    Adds a new observation to an existing stack of observations from previous time-steps.
    :param curr_stack: The current observations stack.
    :param observation: The new observation
    :param stack_size: The required stack size
    :return: The updated observation stack
    """

    if curr_stack == []:
        # starting an episode
        curr_stack = np.vstack(np.expand_dims([observation] * stack_size, 0))
        curr_stack = switch_axes_order(curr_stack, from_type='channels_first', to_type='channels_last')
    else:
        curr_stack = np.append(curr_stack, np.expand_dims(np.squeeze(observation), axis=-1), axis=-1)
        curr_stack = np.delete(curr_stack, 0, -1)

    return curr_stack


def call_method_for_all(instances: List, method: str, args=[], kwargs={}) -> List:
    """
    Calls the same function for all the class instances in the group
    :param instances: a list of class instances to apply the method on
    :param method: the name of the function to be called
    :param args: the positional parameters of the method
    :param kwargs: the named parameters of the method
    :return: a list of the returns values for all the instances
    """
    result = []
    if not isinstance(args, list):
        args = [args]
    sub_methods = method.split('.')  # we allow calling an internal method such as "as_level_manager.train"
    for instance in instances:
        sub_instance = instance
        for sub_method in sub_methods:
            if not hasattr(sub_instance, sub_method):
                raise ValueError("The requested instance method {} does not exist for {}"
                                 .format(sub_method, '.'.join([str(instance.__class__.__name__)] + sub_methods)))
            sub_instance = getattr(sub_instance, sub_method)
        result.append(sub_instance(*args, **kwargs))
    return result


def set_member_values_for_all(instances: List, member: str, val) -> None:
    """
    Calls the same function for all the class instances in the group
    :param instances: a list of class instances to apply the method on
    :param member: the name of the member to be changed
    :param val: the new value to assign
    :return: None
    """
    for instance in instances:
        if not hasattr(instance, member):
            raise ValueError("The requested instance member does not exist")
        setattr(instance, member, val)


def short_dynamic_import(module_path_and_attribute: str, ignore_module_case: bool=False):
    """
    Import by "path:attribute"
    :param module_path_and_attribute: a path to a python file (using dots to separate dirs), followed by a ":" and
                                      an attribute name to import from the path
    :return: the requested attribute
    """
    if '/' in module_path_and_attribute:
        """
        Imports a class from a module using the full path of the module. The path should be given as:
        <full absolute module path with / including .py>:<class name to import>
        And this will be the same as doing "from <full absolute module path> import <class name to import>"
        """
        return dynamic_import_from_full_path(*module_path_and_attribute.split(':'),
                                             ignore_module_case=ignore_module_case)
    else:
        """
        Imports a class from a module using the relative path of the module. The path should be given as:
        <full absolute module path with . and not including .py>:<class name to import>
        And this will be the same as doing "from <full relative module path> import <class name to import>"
        """
        return dynamic_import(*module_path_and_attribute.split(':'),
                              ignore_module_case=ignore_module_case)


def dynamic_import(module_path: str, class_name: str, ignore_module_case: bool=False):
    if ignore_module_case:
        module_name = module_path.split(".")[-1]
        available_modules = os.listdir(os.path.dirname(module_path.replace('.', '/')))
        for module in available_modules:
            curr_module_ext = module.split('.')[-1].lower()
            curr_module_name = module.split('.')[0]
            if curr_module_ext == "py" and curr_module_name.lower() == module_name.lower():
                module_path = '.'.join(module_path.split(".")[:-1] + [curr_module_name])
    module = importlib.import_module(module_path)
    class_ref = getattr(module, class_name)
    return class_ref


def dynamic_import_from_full_path(module_path: str, class_name: str, ignore_module_case: bool=False):
    if ignore_module_case:
        module_name = module_path.split("/")[-1]
        available_modules = os.listdir(os.path.dirname(module_path))
        for module in available_modules:
            curr_module_ext = module.split('.')[-1].lower()
            curr_module_name = module.split('.')[0]
            if curr_module_ext == "py" and curr_module_name.lower() == module_name.lower():
                module_path = '.'.join(module_path.split("/")[:-1] + [curr_module_name])
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_ref = getattr(module, class_name)
    return class_ref


def dynamic_import_and_instantiate_module_from_params(module_parameters, path=None, positional_args=[],
                                                      extra_kwargs={}):
    """
    A function dedicated for coach modules like memory, exploration policy, etc.
    Given the module parameters, it imports it and instantiates it.
    :param module_parameters:
    :return:
    """
    import inspect
    if path is None:
        path = module_parameters.path
    module = short_dynamic_import(path)
    args = set(inspect.getfullargspec(module).args).intersection(module_parameters.__dict__)
    args = {k: module_parameters.__dict__[k] for k in args}
    args = {**args, **extra_kwargs}
    return short_dynamic_import(path)(*positional_args, **args)


def last_sample(state):
    """
    given a batch of states, return the last sample of the batch with length 1
    batch axis.
    """
    return {
        k: np.expand_dims(v[-1], 0)
        for k, v in state.items()
    }


def get_all_subclasses(cls):
    if len(cls.__subclasses__()) == 0:
        return []
    ret = []
    for drv in cls.__subclasses__():
        ret.append(drv)
        ret.extend(get_all_subclasses(drv))

    return ret


class SharedMemoryScratchPad(object):
    def __init__(self):
        self.dict = {}

    def add(self, key, value):
        self.dict[key] = value

    def get(self, key, timeout=30):
        start_time = time.time()
        timeout_passed = False
        while key not in self.dict and not timeout_passed:
            time.sleep(0.1)
            timeout_passed = (time.time() - start_time) > timeout

        if timeout_passed:
            return None
        return self.dict[key]

    def internal_call(self, key, func, args: Tuple):
        if type(args) != tuple:
            args = (args,)
        return getattr(self.dict[key], func)(*args)


class Timer(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(self.prefix, time.time() - self.start)


class ReaderWriterLock(object):
    def __init__(self):
        self.num_readers_lock = Manager().Lock()
        self.writers_lock = Manager().Lock()
        self.num_readers = 0
        self.now_writing = False

    def some_worker_is_reading(self):
        return self.num_readers > 0

    def some_worker_is_writing(self):
        return self.now_writing is True

    def lock_writing_and_reading(self):
        self.writers_lock.acquire()  # first things first - block all other writers
        self.now_writing = True  # block new readers who haven't started reading yet
        while self.some_worker_is_reading():  # let existing readers finish their homework
            time.sleep(0.05)

    def release_writing_and_reading(self):
        self.now_writing = False  # release readers - guarantee no readers starvation
        self.writers_lock.release()  # release writers

    def lock_writing(self):
        while self.now_writing:
            time.sleep(0.05)

        self.num_readers_lock.acquire()
        self.num_readers += 1
        self.num_readers_lock.release()

    def release_writing(self):
        self.num_readers_lock.acquire()
        self.num_readers -= 1
        self.num_readers_lock.release()


class ProgressBar(object):
    def __init__(self, max_value):
        self.start_time = time.time()
        self.max_value = max_value
        self.current_value = 0

    def update(self, current_value, additional_info=""):
        self.current_value = current_value
        percentage = int((100 * current_value) / self.max_value)
        sys.stdout.write("\rProgress: ({}/{}) Time: {} sec {}%|{}{}|  {}"
                         .format(current_value, self.max_value,
                                 round(time.time() - self.start_time, 2),
                                 percentage, '#' * int(percentage / 10),
                                 ' ' * (10 - int(percentage / 10)),
                                 additional_info))
        sys.stdout.flush()

    def close(self):
        print("")


def start_shell_command_and_wait(command):
    p = Popen(command, shell=True, preexec_fn=os.setsid)

    def cleanup():
        os.killpg(os.getpgid(p.pid), 15)

    atexit.register(cleanup)
    p.wait()
    atexit.unregister(cleanup)


def indent_string(string):
    return '\t' + string.replace('\n', '\n\t')
