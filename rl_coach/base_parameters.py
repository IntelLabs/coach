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

from rl_coach.core_types import TrainingSteps, EnvironmentSteps, GradientClippingMethod, RunPhase, \
    SelectedPhaseOnlyDumpFilter, MaxDumpFilter
from rl_coach.filters.filter import NoInputFilter


class Frameworks(Enum):
    tensorflow = "TensorFlow"
    mxnet = "MXNet"


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

class RunType(Enum):
    ORCHESTRATOR = "orchestrator"
    TRAINER = "trainer"
    ROLLOUT_WORKER = "rollout-worker"

    def __str__(self):
        return self.value


class DeviceType(Enum):
    CPU = 'cpu'
    GPU = 'gpu'


class Device(object):
    def __init__(self, device_type: DeviceType, index: int=0):
        """
        :param device_type: type of device (CPU/GPU)
        :param index: index of device (only used if device type is GPU)
        """
        self._device_type = device_type
        self._index = index

    @property
    def device_type(self):
        return self._device_type

    @property
    def index(self):
        return self._index

    def __str__(self):
        return "{}{}".format(self._device_type, self._index)

    def __repr__(self):
        return str(self)


# DistributedCoachSynchronizationType provides the synchronization type for distributed Coach.
# The default value is None, which means the algorithm or preset cannot be used with distributed Coach.
class DistributedCoachSynchronizationType(Enum):
    # In SYNC mode, the trainer waits for all the experiences to be gathered from distributed rollout workers before
    # training a new policy and the rollout workers wait for a new policy before gathering experiences.
    SYNC = "sync"

    # In ASYNC mode, the trainer doesn't wait for any set of experiences to be gathered from distributed rollout workers
    # and the rollout workers continously gather experiences loading new policies, whenever they become available.
    ASYNC = "async"


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
        self.store_transitions_only_when_episodes_are_terminated = False

        # HRL / HER related params
        self.in_action_space = None

        # distributed agents params
        self.share_statistics_between_workers = True

        # intrinsic reward
        self.scale_external_reward_by_intrinsic_reward_value = False

        # n-step returns
        self.n_step = -1  # calculate the total return (no bootstrap, by default)

        # Distributed Coach params
        self.distributed_coach_synchronization_type = None

        # Should the workers wait for full episode
        self.act_for_full_episodes = False


class PresetValidationParameters(Parameters):
    def __init__(self,
                 test=False,
                 min_reward_threshold=0,
                 max_episodes_to_achieve_reward=1,
                 num_workers=1,
                 reward_test_level=None,
                 test_using_a_trace_test=True,
                 trace_test_levels=None,
                 trace_max_env_steps=5000):
        """
        :param test:
            A flag which specifies if the preset should be tested as part of the validation process.
        :param min_reward_threshold:
            The minimum reward that the agent should pass after max_episodes_to_achieve_reward episodes when the
            preset is run.
        :param max_episodes_to_achieve_reward:
            The maximum number of episodes that the agent should train using the preset in order to achieve the
            reward specified by min_reward_threshold.
        :param num_workers:
            The number of workers that should be used when running this preset in the test suite for validation.
        :param reward_test_level:
            The environment level or levels, given by a list of strings, that should be tested as part of the
            reward tests suite.
        :param test_using_a_trace_test:
            A flag that specifies if the preset should be run as part of the trace tests suite.
        :param trace_test_levels:
            The environment level or levels, given by a list of strings, that should be tested as part of the
            trace tests suite.
        :param trace_max_env_steps:
            An integer representing the maximum number of environment steps to run when running this preset as part
            of the trace tests suite.
        """
        super().__init__()

        # setting a seed will only work for non-parallel algorithms. Parallel algorithms add uncontrollable noise in
        # the form of different workers starting at different times, and getting different assignments of CPU
        # time from the OS.

        # Testing parameters
        self.test = test
        self.min_reward_threshold = min_reward_threshold
        self.max_episodes_to_achieve_reward = max_episodes_to_achieve_reward
        self.num_workers = num_workers
        self.reward_test_level = reward_test_level
        self.test_using_a_trace_test = test_using_a_trace_test
        self.trace_test_levels = trace_test_levels
        self.trace_max_env_steps = trace_max_env_steps


class NetworkParameters(Parameters):
    def __init__(self,
                 force_cpu=False,
                 async_training=False,
                 shared_optimizer=True,
                 scale_down_gradients_by_number_of_workers_for_sync_training=True,
                 clip_gradients=None,
                 gradients_clipping_method=GradientClippingMethod.ClipByGlobalNorm,
                 l2_regularization=0,
                 learning_rate=0.00025,
                 learning_rate_decay_rate=0,
                 learning_rate_decay_steps=0,
                 input_embedders_parameters={},
                 embedding_merger_type=EmbeddingMergerType.Concat,
                 middleware_parameters=None,
                 heads_parameters=[],
                 use_separate_networks_per_head=False,
                 optimizer_type='Adam',
                 optimizer_epsilon=0.0001,
                 adam_optimizer_beta1=0.9,
                 adam_optimizer_beta2=0.99,
                 rms_prop_optimizer_decay=0.9,
                 batch_size=32,
                 replace_mse_with_huber_loss=False,
                 create_target_network=False,
                 tensorflow_support=True):
        """
        :param force_cpu:
            Force the neural networks to run on the CPU even if a GPU is available
        :param async_training:
            If set to True, asynchronous training will be used, meaning that each workers will progress in its own
            speed, while not waiting for the rest of the workers to calculate their gradients.
        :param shared_optimizer:
            If set to True, a central optimizer which will be shared with all the workers will be used for applying
            gradients to the network. Otherwise, each worker will have its own optimizer with its own internal
            parameters that will only be affected by the gradients calculated by that worker
        :param scale_down_gradients_by_number_of_workers_for_sync_training:
            If set to True, in synchronous training, the gradients of each worker will be scaled down by the
            number of workers. This essentially means that the gradients applied to the network are the average
            of the gradients over all the workers.
        :param clip_gradients:
            A value that will be used for clipping the gradients of the network. If set to None, no gradient clipping
            will be applied. Otherwise, the gradients will be clipped according to the gradients_clipping_method.
        :param gradients_clipping_method:
            A gradient clipping method, defined by a GradientClippingMethod enum, and that will be used to clip the
            gradients of the network. This will only be used if the clip_gradients value is defined as a value other
            than None.
        :param l2_regularization:
            A L2 regularization weight that will be applied to the network weights while calculating the loss function
        :param learning_rate:
            The learning rate for the network
        :param learning_rate_decay_rate:
            If this value is larger than 0, an exponential decay will be applied to the network learning rate.
            The rate of the decay is defined by this parameter, and the number of training steps the decay will be
            applied is defined by learning_rate_decay_steps. Notice that both parameters should be defined in order
            for this to work correctly.
        :param learning_rate_decay_steps:
            If the learning_rate_decay_rate of the network is larger than 0, an exponential decay will be applied to
            the network learning rate. The number of steps the decay will be applied is defined by this parameter.
            Notice that both this parameter, as well as learning_rate_decay_rate should be defined in order for the
            learning rate decay to work correctly.
        :param input_embedders_parameters:
            A dictionary mapping between input names and input embedders (InputEmbedderParameters) to use for the
            network. Each of the keys is an input name as returned from the environment in the state.
            For example, if the environment returns a state containing 'observation' and 'measurements', then
            the keys for the input embedders dictionary can be either 'observation' to use the observation as input,
            'measurements' to use the measurements as input, or both.
            The embedder type will be automatically selected according to the input type. Vector inputs will
            produce a fully connected embedder, and image inputs will produce a convolutional embedder.
        :param embedding_merger_type:
            The type of embedding merging to use, given by one of the EmbeddingMergerType enum values.
            This will be used to merge the outputs of all the input embedders into a single embbeding.
        :param middleware_parameters:
            The parameters of the middleware to use, given by a MiddlewareParameters object.
            Each network will have only a single middleware embedder which will take the merged embeddings from the
            input embedders and pass them through more neural network layers.
        :param heads_parameters:
            A list of heads for the network given by their corresponding HeadParameters.
            Each network can have one or multiple network heads, where each one will take the output of the middleware
            and make some additional computation on top of it. Additionally, each head calculates a weighted loss value,
            and the loss values from all the heads will be summed later on.
        :param use_separate_networks_per_head:
            A flag that allows using different copies of the input embedders and middleware for each one of the heads.
            Regularly, the heads will have a shared input, but in the case where use_separate_networks_per_head is set
            to True, each one of the heads will get a different input.
        :param optimizer_type:
            A string specifying the optimizer type to use for updating the network. The available optimizers are
            Adam, RMSProp and LBFGS.
        :param optimizer_epsilon:
            An internal optimizer parameter used for Adam and RMSProp.
        :param adam_optimizer_beta1:
            An beta1 internal optimizer parameter used for Adam. It will be used only if Adam was selected as the
            optimizer for the network.
        :param adam_optimizer_beta2:
            An beta2 internal optimizer parameter used for Adam. It will be used only if Adam was selected as the
            optimizer for the network.
        :param rms_prop_optimizer_decay:
            The decay value for the RMSProp optimizer, which will be used only in case the RMSProp optimizer was
            selected for this network.
        :param batch_size:
            The batch size to use when updating the network.
        :param replace_mse_with_huber_loss:
        :param create_target_network:
            If this flag is set to True, an additional copy of the network will be created and initialized with the
            same weights as the online network. It can then be queried, and its weights can be synced from the
            online network at will.
        :param tensorflow_support:
            A flag which specifies if the network is supported by the TensorFlow framework.
        """
        super().__init__()
        self.framework = Frameworks.tensorflow
        self.sess = None

        # hardware parameters
        self.force_cpu = force_cpu

        # distributed training options
        self.async_training = async_training
        self.shared_optimizer = shared_optimizer
        self.scale_down_gradients_by_number_of_workers_for_sync_training = scale_down_gradients_by_number_of_workers_for_sync_training

        # regularization
        self.clip_gradients = clip_gradients
        self.gradients_clipping_method = gradients_clipping_method
        self.l2_regularization = l2_regularization

        # learning rate
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps

        # structure
        self.input_embedders_parameters = input_embedders_parameters
        self.embedding_merger_type = embedding_merger_type
        self.middleware_parameters = middleware_parameters
        self.heads_parameters = heads_parameters
        self.use_separate_networks_per_head = use_separate_networks_per_head
        self.optimizer_type = optimizer_type
        self.optimizer_epsilon = optimizer_epsilon
        self.adam_optimizer_beta1 = adam_optimizer_beta1
        self.adam_optimizer_beta2 = adam_optimizer_beta2
        self.rms_prop_optimizer_decay = rms_prop_optimizer_decay
        self.batch_size = batch_size
        self.replace_mse_with_huber_loss = replace_mse_with_huber_loss
        self.create_target_network = create_target_network

        # Framework support
        self.tensorflow_support = tensorflow_support


class NetworkComponentParameters(Parameters):
    def __init__(self, dense_layer):
        self.dense_layer = dense_layer


class VisualizationParameters(Parameters):
    def __init__(self,
                 print_networks_summary=False,
                 dump_csv=True,
                 dump_signals_to_csv_every_x_episodes=5,
                 dump_gifs=False,
                 dump_mp4=False,
                 video_dump_methods=None,
                 dump_in_episode_signals=False,
                 dump_parameters_documentation=True,
                 render=False,
                 native_rendering=False,
                 max_fps_for_human_control=10,
                 tensorboard=False,
                 add_rendered_image_to_env_response=False):
        """
        :param print_networks_summary:
            If set to True, a summary of all the networks structure will be printed at the beginning of the experiment
        :param dump_csv:
            If set to True, the logger will dump logs to a csv file once in every dump_signals_to_csv_every_x_episodes
            episodes. The logs can be later used to visualize the training process using Coach Dashboard.
        :param dump_signals_to_csv_every_x_episodes:
            Defines the number of episodes between writing new data to the csv log files. Lower values can affect
            performance, as writing to disk may take time, and it is done synchronously.
        :param dump_gifs:
            If set to True, GIF videos of the environment will be stored into the experiment directory according to
            the filters defined in video_dump_methods.
        :param dump_mp4:
            If set to True, MP4 videos of the environment will be stored into the experiment directory according to
            the filters defined in video_dump_methods.
        :param dump_in_episode_signals:
            If set to True, csv files will be dumped for each episode for inspecting different metrics within the
            episode. This means that for each step in each episode, different metrics such as the reward, the
            future return, etc. will be saved. Setting this to True may affect performance severely, and therefore
            this should be used only for debugging purposes.
        :param dump_parameters_documentation:
            If set to True, a json file containing all the agent parameters will be saved in the experiment directory.
            This may be very useful for inspecting the values defined for each parameters and making sure that all
            the parameters are defined as expected.
        :param render:
            If set to True, the environment render function will be called for each step, rendering the image of the
            environment. This may affect the performance of training, and is highly dependent on the environment.
            By default, Coach uses PyGame to render the environment image instead of the environment specific rendered.
            To change this, use the native_rendering flag.
        :param native_rendering:
            If set to True, the environment native renderer will be used for rendering the environment image.
            In some cases this can be slower than rendering using PyGame through Coach, but in other cases the
            environment opens its native renderer by default, so rendering with PyGame is an unnecessary overhead.
        :param max_fps_for_human_control:
            The maximum number of frames per second used while playing the environment as a human. This only has
            effect while using the --play flag for Coach.
        :param tensorboard:
            If set to True, TensorBoard summaries will be stored in the experiment directory. This can later be
            loaded in TensorBoard in order to visualize the training process.
        :param video_dump_methods:
            A list of dump methods that will be used as filters for deciding when to save videos.
            The filters in the list will be checked one after the other until the first dump method that returns
            false for should_dump() in the environment class. This list will only be used if dump_mp4 or dump_gif are
            set to True.
        :param add_rendered_image_to_env_response:
            Some environments have a different observation compared to the one displayed while rendering.
            For some cases it can be useful to pass the rendered image to the agent for visualization purposes.
            If this flag is set to True, the rendered image will be added to the environment EnvResponse object,
            which will be passed to the agent and allow using those images.
        """
        super().__init__()
        if video_dump_methods is None:
            video_dump_methods = [SelectedPhaseOnlyDumpFilter(RunPhase.TEST), MaxDumpFilter()]
        self.print_networks_summary = print_networks_summary
        self.dump_csv = dump_csv
        self.dump_gifs = dump_gifs
        self.dump_mp4 = dump_mp4
        self.dump_signals_to_csv_every_x_episodes = dump_signals_to_csv_every_x_episodes
        self.dump_in_episode_signals = dump_in_episode_signals
        self.dump_parameters_documentation = dump_parameters_documentation
        self.render = render
        self.native_rendering = native_rendering
        self.max_fps_for_human_control = max_fps_for_human_control
        self.tensorboard = tensorboard
        self.video_dump_filters = video_dump_methods
        self.add_rendered_image_to_env_response = add_rendered_image_to_env_response


class AgentParameters(Parameters):
    def __init__(self, algorithm: AlgorithmParameters, exploration: 'ExplorationParameters', memory: 'MemoryParameters',
                 networks: Dict[str, NetworkParameters], visualization: VisualizationParameters=VisualizationParameters()):
        """
        :param algorithm:
            A class inheriting AlgorithmParameters.
            The parameters used for the specific algorithm used by the agent.
            These parameters can be later referenced in the agent implementation through self.ap.algorithm.
        :param exploration:
            Either a class inheriting ExplorationParameters or a dictionary mapping between action
            space types and their corresponding ExplorationParameters. If a dictionary was used,
            when the agent will be instantiated, the correct exploration policy parameters will be used
            according to the real type of the environment action space.
            These parameters will be used to instantiate the exporation policy.
        :param memory:
            A class inheriting MemoryParameters. It defines all the parameters used by the memory module.
        :param networks:
            A dictionary mapping between network names and their corresponding network parmeters, defined
            as a class inheriting NetworkParameters. Each element will be used in order to instantiate
            a NetworkWrapper class, and all the network wrappers will be stored in the agent under
            self.network_wrappers. self.network_wrappers is a dict mapping between the network name that
            was given in the networks dict, and the instantiated network wrapper.
        :param visualization:
            A class inheriting VisualizationParameters and defining various parameters that can be
            used for visualization purposes, such as printing to the screen, rendering, and saving videos.
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
        self.full_name_id = None
        self.name = None
        self.is_a_highest_level_agent = True
        self.is_a_lowest_level_agent = True
        self.task_parameters = None

    @property
    def path(self):
        return 'rl_coach.agents.agent:Agent'


class TaskParameters(Parameters):
    def __init__(self, framework_type: Frameworks=Frameworks.tensorflow, evaluate_only: bool=False, use_cpu: bool=False,
                 experiment_path='/tmp', seed=None, checkpoint_save_secs=None, checkpoint_restore_dir=None,
                 checkpoint_save_dir=None, export_onnx_graph: bool=False, apply_stop_condition: bool=False,
                 num_gpu: int=1):
        """
        :param framework_type: deep learning framework type. currently only tensorflow is supported
        :param evaluate_only: the task will be used only for evaluating the model
        :param use_cpu: use the cpu for this task
        :param experiment_path: the path to the directory which will store all the experiment outputs
        :param seed: a seed to use for the random numbers generator
        :param checkpoint_save_secs: the number of seconds between each checkpoint saving
        :param checkpoint_restore_dir: the directory to restore the checkpoints from
        :param checkpoint_save_dir: the directory to store the checkpoints in
        :param export_onnx_graph: If set to True, this will export an onnx graph each time a checkpoint is saved
        :param apply_stop_condition: If set to True, this will apply the stop condition defined by reaching a target success rate
        :param num_gpu: number of GPUs to use
        """
        self.framework_type = framework_type
        self.task_index = 0  # TODO: not really needed
        self.evaluate_only = evaluate_only
        self.use_cpu = use_cpu
        self.experiment_path = experiment_path
        self.checkpoint_save_secs = checkpoint_save_secs
        self.checkpoint_restore_dir = checkpoint_restore_dir
        self.checkpoint_save_dir = checkpoint_save_dir
        self.seed = seed
        self.export_onnx_graph = export_onnx_graph
        self.apply_stop_condition = apply_stop_condition
        self.num_gpu = num_gpu


class DistributedTaskParameters(TaskParameters):
    def __init__(self, framework_type: Frameworks, parameters_server_hosts: str, worker_hosts: str, job_type: str,
                 task_index: int, evaluate_only: bool=False, num_tasks: int=None,
                 num_training_tasks: int=None, use_cpu: bool=False, experiment_path=None, dnd=None,
                 shared_memory_scratchpad=None, seed=None, checkpoint_save_secs=None, checkpoint_restore_dir=None,
                 checkpoint_save_dir=None, export_onnx_graph: bool=False, apply_stop_condition: bool=False):
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
        :param checkpoint_save_secs: the number of seconds between each checkpoint saving
        :param checkpoint_restore_dir: the directory to restore the checkpoints from
        :param checkpoint_save_dir: the directory to store the checkpoints in
        :param export_onnx_graph: If set to True, this will export an onnx graph each time a checkpoint is saved
        :param apply_stop_condition: If set to True, this will apply the stop condition defined by reaching a target success rate

        """
        super().__init__(framework_type=framework_type, evaluate_only=evaluate_only, use_cpu=use_cpu,
                         experiment_path=experiment_path, seed=seed, checkpoint_save_secs=checkpoint_save_secs,
                         checkpoint_restore_dir=checkpoint_restore_dir, checkpoint_save_dir=checkpoint_save_dir,
                         export_onnx_graph=export_onnx_graph, apply_stop_condition=apply_stop_condition)
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
