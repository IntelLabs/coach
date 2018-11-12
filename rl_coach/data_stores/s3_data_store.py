from rl_coach.data_stores.data_store import DataStore, DataStoreParameters
from kubernetes import client as k8sclient
from minio import Minio
from minio.error import ResponseError
from configparser import ConfigParser, Error
from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from rl_coach.data_stores.data_store import SyncFiles

import os
import time
import io


class S3DataStoreParameters(DataStoreParameters):
    def __init__(self, ds_params, creds_file: str = None, end_point: str = None, bucket_name: str = None,
                 checkpoint_dir: str = None):

        super().__init__(ds_params.store_type, ds_params.orchestrator_type, ds_params.orchestrator_params)
        self.creds_file = creds_file
        self.end_point = end_point
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir


class S3DataStore(DataStore):
    def __init__(self, params: S3DataStoreParameters):
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
        try:
            self.mc.remove_object(self.params.bucket_name, SyncFiles.LOCKFILE.value)

            self.mc.put_object(self.params.bucket_name, SyncFiles.LOCKFILE.value, io.BytesIO(b''), 0)

            checkpoint_file = None
            for root, dirs, files in os.walk(self.params.checkpoint_dir):
                for filename in files:
                    if filename == 'checkpoint':
                        checkpoint_file = (root, filename)
                        continue
                    abs_name = os.path.abspath(os.path.join(root, filename))
                    rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
                    self.mc.fput_object(self.params.bucket_name, rel_name, abs_name)

            abs_name = os.path.abspath(os.path.join(checkpoint_file[0], checkpoint_file[1]))
            rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
            self.mc.fput_object(self.params.bucket_name, rel_name, abs_name)

            self.mc.remove_object(self.params.bucket_name, SyncFiles.LOCKFILE.value)

        except ResponseError as e:
            print("Got exception: %s\n while saving to S3", e)

    def load_from_store(self):
        try:
            filename = os.path.abspath(os.path.join(self.params.checkpoint_dir, "checkpoint"))

            while True:
                objects = self.mc.list_objects_v2(self.params.bucket_name, SyncFiles.LOCKFILE.value)

                if next(objects, None) is None:
                    try:
                        self.mc.fget_object(self.params.bucket_name, "checkpoint", filename)
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

            ckpt = CheckpointState()
            if os.path.exists(filename):
                contents = open(filename, 'r').read()
                text_format.Merge(contents, ckpt)
                rel_path = os.path.relpath(ckpt.model_checkpoint_path, self.params.checkpoint_dir)

                objects = self.mc.list_objects_v2(self.params.bucket_name, prefix=rel_path, recursive=True)
                for obj in objects:
                    filename = os.path.abspath(os.path.join(self.params.checkpoint_dir, obj.object_name))
                    self.mc.fget_object(obj.bucket_name, obj.object_name, filename)

        except ResponseError as e:
            print("Got exception: %s\n while loading from S3", e)
