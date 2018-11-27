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


from rl_coach.data_stores.data_store import DataStore, DataStoreParameters
from minio import Minio
from minio.error import ResponseError
from configparser import ConfigParser, Error
from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.data_stores.data_store import SyncFiles

import os
import time
import io


class S3DataStoreParameters(DataStoreParameters):
    def __init__(self, ds_params, creds_file: str = None, end_point: str = None, bucket_name: str = None,
                 checkpoint_dir: str = None, expt_dir: str = None):

        super().__init__(ds_params.store_type, ds_params.orchestrator_type, ds_params.orchestrator_params)
        self.creds_file = creds_file
        self.end_point = end_point
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir
        self.expt_dir = expt_dir


class S3DataStore(DataStore):
    """
    An implementation of the data store using S3 for storing policy checkpoints when using Coach in distributed mode.
    The policy checkpoints are written by the trainer and read by the rollout worker.
    """

    def __init__(self, params: S3DataStoreParameters):
        """
        :param params: The parameters required to use the S3 data store.
        """

        super(S3DataStore, self).__init__(params)
        self.params = params
        access_key = None
        secret_key = None
        if params.creds_file:
            config = ConfigParser()
            config.read(params.creds_file)
            try:
                access_key = config.get('default', 'aws_access_key_id')
                secret_key = config.get('default', 'aws_secret_access_key')
            except Error as e:
                print("Error when reading S3 credentials file: %s", e)
        else:
            access_key = os.environ.get('ACCESS_KEY_ID')
            secret_key = os.environ.get('SECRET_ACCESS_KEY')
        self.mc = Minio(self.params.end_point, access_key=access_key, secret_key=secret_key)

    def deploy(self) -> bool:
        return True

    def get_info(self):
        return "s3://{}/{}".format(self.params.bucket_name)

    def undeploy(self) -> bool:
        return True

    def save_to_store(self):
        """
        save_to_store() uploads the policy checkpoint, gifs and videos to the S3 data store. It reads the checkpoint state files and
        uploads only the latest checkpoint files to S3. It is used by the trainer in Coach when used in the distributed mode.
        """
        try:
            # remove lock file if it exists
            self.mc.remove_object(self.params.bucket_name, SyncFiles.LOCKFILE.value)

            # Acquire lock
            self.mc.put_object(self.params.bucket_name, SyncFiles.LOCKFILE.value, io.BytesIO(b''), 0)

            state_file = CheckpointStateFile(os.path.abspath(self.params.checkpoint_dir))
            if state_file.exists():
                ckpt_state = state_file.read()
                checkpoint_file = None
                for root, dirs, files in os.walk(self.params.checkpoint_dir):
                    for filename in files:
                        if filename == CheckpointStateFile.checkpoint_state_filename:
                            checkpoint_file = (root, filename)
                            continue
                        if filename.startswith(ckpt_state.name):
                            abs_name = os.path.abspath(os.path.join(root, filename))
                            rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
                            self.mc.fput_object(self.params.bucket_name, rel_name, abs_name)

                abs_name = os.path.abspath(os.path.join(checkpoint_file[0], checkpoint_file[1]))
                rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
                self.mc.fput_object(self.params.bucket_name, rel_name, abs_name)

            # release lock
            self.mc.remove_object(self.params.bucket_name, SyncFiles.LOCKFILE.value)

            if self.params.expt_dir and os.path.exists(self.params.expt_dir):
                for filename in os.listdir(self.params.expt_dir):
                    if filename.endswith((".csv", ".json")):
                        self.mc.fput_object(self.params.bucket_name, filename, os.path.join(self.params.expt_dir, filename))

            if self.params.expt_dir and os.path.exists(os.path.join(self.params.expt_dir, 'videos')):
                for filename in os.listdir(os.path.join(self.params.expt_dir, 'videos')):
                        self.mc.fput_object(self.params.bucket_name, filename, os.path.join(self.params.expt_dir, 'videos', filename))

            if self.params.expt_dir and os.path.exists(os.path.join(self.params.expt_dir, 'gifs')):
                for filename in os.listdir(os.path.join(self.params.expt_dir, 'gifs')):
                        self.mc.fput_object(self.params.bucket_name, filename, os.path.join(self.params.expt_dir, 'gifs', filename))
        except ResponseError as e:
            print("Got exception: %s\n while saving to S3", e)

    def load_from_store(self):
        """
        load_from_store() downloads a new checkpoint from the S3 data store when it is not available locally. It is used
        by the rollout workers when using Coach in distributed mode.
        """
        try:
            state_file = CheckpointStateFile(os.path.abspath(self.params.checkpoint_dir))

            # wait until lock is removed
            while True:
                objects = self.mc.list_objects_v2(self.params.bucket_name, SyncFiles.LOCKFILE.value)

                if next(objects, None) is None:
                    try:
                        # fetch checkpoint state file from S3
                        self.mc.fget_object(self.params.bucket_name, state_file.filename, state_file.path)
                    except Exception as e:
                        continue
                    break
                time.sleep(10)

            # Check if there's a finished file
            objects = self.mc.list_objects_v2(self.params.bucket_name, SyncFiles.FINISHED.value)

            if next(objects, None) is not None:
                try:
                    self.mc.fget_object(
                        self.params.bucket_name, SyncFiles.FINISHED.value,
                        os.path.abspath(os.path.join(self.params.checkpoint_dir, SyncFiles.FINISHED.value))
                    )
                except Exception as e:
                    pass

            checkpoint_state = state_file.read()
            if checkpoint_state is not None:
                objects = self.mc.list_objects_v2(self.params.bucket_name, prefix=checkpoint_state.name, recursive=True)
                for obj in objects:
                    filename = os.path.abspath(os.path.join(self.params.checkpoint_dir, obj.object_name))
                    if not os.path.exists(filename):
                        self.mc.fget_object(obj.bucket_name, obj.object_name, filename)

        except ResponseError as e:
            print("Got exception: %s\n while loading from S3", e)
