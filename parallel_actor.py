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

import argparse
import tensorflow as tf
from architectures import *
from environments import *
from agents import *
from utils import *
import time
import copy
from logger import *
from configurations import *
from presets import *
import shutil

start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps_hosts',
                        help="(string) Comma-separated list of hostname:port pairs",
                        default='',
                        type=str)
    parser.add_argument('--worker_hosts',
                        help="(string) Comma-separated list of hostname:port pairs",
                        default='',
                        type=str)
    parser.add_argument('--job_name',
                        help="(string) One of 'ps', 'worker'",
                        default='',
                        type=str)
    parser.add_argument('--load_json_path',
                        help="(string) Path to a JSON file to load.",
                        default='',
                        type=str)

    args = parser.parse_args()

    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    if args.job_name == "ps":
        # Create and start a parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=0,
                                 config=tf.ConfigProto())#device_filters=["/job:ps"]))
        server.join()

    elif args.job_name == "worker":
        # get tuning parameters
        tuning_parameters = json_to_preset(args.load_json_path)

        # dump documentation
        if not os.path.exists(tuning_parameters.experiment_path):
            os.makedirs(tuning_parameters.experiment_path)
        if tuning_parameters.evaluate_only:
            logger.set_dump_dir(tuning_parameters.experiment_path, tuning_parameters.task_id, filename='evaluator')
        else:
            logger.set_dump_dir(tuning_parameters.experiment_path, tuning_parameters.task_id)

        # multi-threading parameters
        tuning_parameters.start_time = start_time

        # User is allowed to override the number of synchronized threads if he wishes to do so.
        # Else, just sync over all of them.
        if not tuning_parameters.synchronize_over_num_threads:
            tuning_parameters.synchronize_over_num_threads = tuning_parameters.num_threads

        tuning_parameters.distributed = True
        if tuning_parameters.evaluate_only:
            tuning_parameters.visualization.dump_signals_to_csv_every_x_episodes = 1

        # Create and start a worker
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=tuning_parameters.task_id)

        # Assigns ops to the local worker by default.
        device = tf.train.replica_device_setter(worker_device="/job:worker/task:%d/cpu:0" % tuning_parameters.task_id,
                                                cluster=cluster)

        # create the agent and the environment
        env_instance = create_environment(tuning_parameters)
        exec('agent = ' + tuning_parameters.agent.type + '(env_instance, tuning_parameters, replicated_device=device, '
                                                    'thread_id=tuning_parameters.task_id)')

        # building the scaffold
        # local vars
        local_variables = []
        for network in agent.networks:
            local_variables += network.get_local_variables()
        local_variables += tf.local_variables()

        # global vars
        global_variables = []
        for network in agent.networks:
            global_variables += network.get_global_variables()

        # out of scope variables - not sure why this variables are created out of scope
        variables_not_in_scope = [v for v in tf.global_variables() if v not in global_variables and v not in local_variables]

        # init ops
        global_init_op = tf.variables_initializer(global_variables)
        local_init_op = tf.variables_initializer(local_variables + variables_not_in_scope)
        out_of_scope_init_op = tf.variables_initializer(variables_not_in_scope)
        init_all_op = tf.global_variables_initializer()  # this includes global, local, and out of scope
        ready_op = tf.report_uninitialized_variables(global_variables + local_variables)
        ready_for_local_init_op = tf.report_uninitialized_variables([])

        def init_fn(scaffold, session):
            session.run(init_all_op)


        #saver = tf.train.Saver(max_to_keep=None) # uncomment to unlimit number of stored checkpoints
        scaffold = tf.train.Scaffold(init_op=init_all_op,
                                     init_fn=init_fn,
                                     ready_op=ready_op,
                                     ready_for_local_init_op=ready_for_local_init_op,
                                     local_init_op=local_init_op)
                                     #saver=saver) # uncomment to unlimit number of stored checkpoints

        # Due to awkward tensorflow behavior where the same variable is used to decide whether to restore a model
        # (and where from), or just save the model (and where to), we employ the below. In case where a restore folder
        # is given, it will also be used as the folder to save new checkpoints of the trained model to. Otherwise the
        # experiment's folder will be used as the folder to save the trained model to.
        if tuning_parameters.checkpoint_restore_dir:
            checkpoint_dir = tuning_parameters.checkpoint_restore_dir
        elif tuning_parameters.save_model_sec:
            checkpoint_dir = tuning_parameters.experiment_path
        else:
            checkpoint_dir = None

        # Set the session
        sess = tf.train.MonitoredTrainingSession(
            server.target,
            is_chief=tuning_parameters.task_id == 0,
            scaffold=scaffold,
            hooks=[],
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_secs=tuning_parameters.save_model_sec)
        tuning_parameters.sess = sess
        for network in agent.networks:
            network.set_session(sess)
            # if hasattr(network.global_network, 'lock_init'):
            #     sess.run(network.global_network.lock_init)
            # if hasattr(network.global_network, 'release_init'):
            #     sess.run(network.global_network.release_init)

        if tuning_parameters.visualization.tensorboard:
            # Write the merged summaries to the current experiment directory
            agent.main_network.online_network.train_writer = tf.summary.FileWriter(
                tuning_parameters.experiment_path + '/tensorboard_worker{}'.format(tuning_parameters.task_id),
                sess.graph)

        # Start the training or evaluation
        if tuning_parameters.evaluate_only:
            agent.evaluate(sys.maxsize, keep_networks_synced=True)  # evaluate forever
        else:
            agent.improve()
    else:
        screen.error("Invalid mode requested for parallel_actor.")
        exit(1)

