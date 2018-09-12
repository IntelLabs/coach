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
import inspect
import json
import os
import sys
import types
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Union

from rl_coach.core_types import TrainingSteps, EnvironmentSteps, GradientClippingMethod
from rl_coach.filters.filter import NoInputFilter


class Frameworks(Enum):
    tensorflow = "TensorFlow"


class EmbedderScheme(Enum):
    Empty = "Empty"
    Shallow = "Shallow"
    Medium = "Medium"
    Deep = "Deep"


class MiddlewareScheme(Enum):
    Empty = "Empty"
    Shallow = "Shallow"
    Medium = "Medium"
    Deep = "Deep"


class EmbeddingMergerType(Enum):
    Concat = 0
    Sum = 1
    #ConcatDepthWise = 2
    #Multiply = 3


def iterable_to_items(obj):
    if isinstance(obj, dict) or isinstance(obj, OrderedDict) or isinstance(obj, types.MappingProxyType):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        raise ValueError("The given object is not a dict or a list")
    return items


def unfold_dict_or_list(obj: Union[Dict, List, OrderedDict]):
    """
    Recursively unfolds all the parameters in dictionaries and lists
    :param obj: a dictionary or list to unfold
    :return: the unfolded parameters dictionary
    """
    parameters = OrderedDict()
    items = iterable_to_items(obj)
    for k, v in items:
        if isinstance(v, dict) or isinstance(v, list) or isinstance(v, OrderedDict):
            if 'tensorflow.' not in str(v.__class__):
                parameters[k] = unfold_dict_or_list(v)
        elif 'tensorflow.' in str(v.__class__):
            parameters[k] = v
        elif hasattr(v, '__dict__'):
            sub_params = v.__dict__
            if '__objclass__' not in sub_params.keys():
                try:
                    parameters[k] = unfold_dict_or_list(sub_params)
                except RecursionError:
                    parameters[k] = sub_params
                parameters[k]['__class__'] = v.__class__.__name__
            else:
                # unfolding this type of object will result in infinite recursion
                parameters[k] = sub_params
        else:
            parameters[k] = v
    if not isinstance(obj, OrderedDict) and not isinstance(obj, list):
        parameters = OrderedDict(sorted(parameters.items()))
    return parameters


class Parameters(object):
    def __setattr__(self, key, value):
        caller_name = sys._getframe(1).f_code.co_name

        if caller_name != '__init__' and not hasattr(self, key):
            raise TypeError("Parameter '{}' does not exist in {}. Parameters are only to be defined in a constructor of"
                            " a class inheriting from Parameters. In order to explicitly register a new parameter "
                            "outside of a constructor use register_var().".
                            format(key, self.__class__))
        object.__setattr__(self, key, value)

    @property
    def path(self):
        if hasattr(self, 'parameterized_class_name'):
            module_path = os.path.relpath(inspect.getfile(self.__class__), os.getcwd())[:-3] + '.py'

            return ':'.join([module_path, self.parameterized_class_name])
        else:
            raise ValueError("The parameters class does not have an attached class it parameterizes. "
                             "The self.parameterized_class_name should be set to the parameterized class.")

    def register_var(self, key, value):
        if hasattr(self, key):
            raise TypeError("Cannot register an already existing parameter '{}'. ".format(key))
        object.__setattr__(self, key, value)

    def __str__(self):
        result = "\"{}\" {}\n".format(self.__class__.__name__,
                                   json.dumps(unfold_dict_or_list(self.__dict__), indent=4, default=repr))
        return result


class AlgorithmParameters(Parameters):
    def __init__(self):
        # Architecture parameters
        self.use_accumulated_reward_as_measurement = False

        # Agent parameters
        self.num_consecutive_playing_steps = EnvironmentSteps(1)
        self.num_consecutive_training_steps = 1  # TODO: update this to TrainingSteps

        self.heatup_using_network_decisions = False
        self.discount = 0.99
        self.apply_gradients_every_x_episodes = 5
        self.num_steps_between_copying_online_weights_to_target = TrainingSteps(0)
        self.rate_for_copying_weights_to_target = 1.0
        self.load_memory_from_file_path = None
        self.collect_new_data = True
        self.store_transitions_only_when_episodes_are_terminated = False

        # HRL / HER related params
        self.in_action_space = None

        # distributed agents params
        self.share_statistics_between_workers = True

        # intrinsic reward
        self.scale_external_reward_by_intrinsic_reward_value = False


class PresetValidationParameters(Parameters):
    def __init__(self):
        super().__init__()

        # setting a seed will only work for non-parallel algorithms. Parallel algorithms add uncontrollable noise in
        # the form of different workers starting at different times, and getting different assignments of CPU
        # time from the OS.

        # Testing parameters
        self.test = False
        self.min_reward_threshold = 0
        self.max_episodes_to_achieve_reward = 1
        self.num_workers = 1
        self.reward_test_level = None
        self.test_using_a_trace_test = True
        self.trace_test_levels = None
        self.trace_max_env_steps = 5000


class NetworkParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.framework = Frameworks.tensorflow
        self.sess = None

        # hardware parameters
        self.force_cpu = False

        # distributed training options
        self.num_threads = 1
        self.synchronize_over_num_threads = 1
        self.distributed = False
        self.async_training = False
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = True

        # regularization
        self.clip_gradients = None
        self.gradients_clipping_method = GradientClippingMethod.ClipByGlobalNorm
        self.kl_divergence_constraint = None
        self.l2_regularization = 0

        # learning rate
        self.learning_rate = 0.00025
        self.learning_rate_decay_rate = 0
        self.learning_rate_decay_steps = 0

        # structure
        self.input_embedders_parameters = {}
        self.embedding_merger_type = EmbeddingMergerType.Concat
        self.middleware_parameters = None
        self.heads_parameters = []
        self.num_output_head_copies = 1
        self.loss_weights = []
        self.rescale_gradient_from_head_by_factor = [1]
        self.use_separate_networks_per_head = False
        self.optimizer_type = 'Adam'
        self.optimizer_epsilon = 0.0001
        self.adam_optimizer_beta1 = 0.9
        self.adam_optimizer_beta2 = 0.99
        self.rms_prop_optimizer_decay = 0.9
        self.batch_size = 32
        self.replace_mse_with_huber_loss = False
        self.create_target_network = False

        # Framework support
        self.tensorflow_support = True


class NetworkComponentParameters(Parameters):
    def __init__(self, dense_layer):
        self.dense_layer = dense_layer


class VisualizationParameters(Parameters):
    def __init__(self):
        super().__init__()
        # Visualization parameters
        self.print_summary = True
        self.dump_csv = True
        self.dump_gifs = False
        self.dump_mp4 = False
        self.dump_signals_to_csv_every_x_episodes = 5
        self.dump_in_episode_signals = False
        self.dump_parameters_documentation = True
        self.render = False
        self.native_rendering = False
        self.max_fps_for_human_control = 10
        self.tensorboard = False
        self.video_dump_methods = []  # a list of dump methods which will be checked one after the other until the first
                                      # dump method that returns false for should_dump()
        self.add_rendered_image_to_env_response = False


class AgentParameters(Parameters):
    def __init__(self, algorithm: AlgorithmParameters, exploration: 'ExplorationParameters', memory: 'MemoryParameters',
                 networks: Dict[str, NetworkParameters], visualization: VisualizationParameters=VisualizationParameters()):
        """
        :param algorithm: the algorithmic parameters
        :param exploration: the exploration policy parameters
        :param memory: the memory module parameters
        :param networks: the parameters for the networks of the agent
        :param visualization: the visualization parameters
        """
        super().__init__()
        self.visualization = visualization
        self.algorithm = algorithm
        self.exploration = exploration
        self.memory = memory
        self.network_wrappers = networks
        self.input_filter = None
        self.output_filter = None
        self.pre_network_filter = NoInputFilter()
        self.full_name_id = None  # TODO: do we really want to hold this parameter here?
        self.name = None
        self.is_a_highest_level_agent = True
        self.is_a_lowest_level_agent = True
        self.task_parameters = None

    @property
    def path(self):
        return 'rl_coach.agents.agent:Agent'


class TaskParameters(Parameters):
    def __init__(self, framework_type: str, evaluate_only: bool=False, use_cpu: bool=False, experiment_path=None,
                 seed=None):
        """
        :param framework_type: deep learning framework type. currently only tensorflow is supported
        :param evaluate_only: the task will be used only for evaluating the model
        :param use_cpu: use the cpu for this task
        :param experiment_path: the path to the directory which will store all the experiment outputs
        :param seed: a seed to use for the random numbers generator
        """
        self.framework_type = framework_type
        self.task_index = None  # TODO: not really needed
        self.evaluate_only = evaluate_only
        self.use_cpu = use_cpu
        self.experiment_path = experiment_path
        self.seed = seed


class DistributedTaskParameters(TaskParameters):
    def __init__(self, framework_type: str, parameters_server_hosts: str, worker_hosts: str, job_type: str,
                 task_index: int, evaluate_only: bool=False, num_tasks: int=None,
                 num_training_tasks: int=None, use_cpu: bool=False, experiment_path=None, dnd=None,
                 shared_memory_scratchpad=None, seed=None):
        """
        :param framework_type: deep learning framework type. currently only tensorflow is supported
        :param evaluate_only: the task will be used only for evaluating the model
        :param parameters_server_hosts: comma-separated list of hostname:port pairs to which the parameter servers are
                                        assigned
        :param worker_hosts: comma-separated list of hostname:port pairs to which the workers are assigned
        :param job_type: the job type - either ps (short for parameters server) or worker
        :param task_index: the index of the process
        :param num_tasks: the number of total tasks that are running (not including the parameters server)
        :param num_training_tasks: the number of tasks that are training (not including the parameters server)
        :param use_cpu: use the cpu for this task
        :param experiment_path: the path to the directory which will store all the experiment outputs
        :param dnd: an external DND to use for NEC. This is a workaround needed for a shared DND not using the scratchpad.
        :param seed: a seed to use for the random numbers generator
        """
        super().__init__(framework_type=framework_type, evaluate_only=evaluate_only, use_cpu=use_cpu,
                         experiment_path=experiment_path, seed=seed)
        self.parameters_server_hosts = parameters_server_hosts
        self.worker_hosts = worker_hosts
        self.job_type = job_type
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.num_training_tasks = num_training_tasks
        self.device = None  # the replicated device which will be used for the global parameters
        self.worker_target = None
        self.dnd = dnd
        self.shared_memory_scratchpad = shared_memory_scratchpad
