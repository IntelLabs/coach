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


from rl_coach.data_stores.nfs_data_store import NFSDataStore, NFSDataStoreParameters
from rl_coach.data_stores.s3_data_store import S3DataStore, S3DataStoreParameters
from rl_coach.data_stores.data_store import DataStoreParameters


def get_data_store(params):
    data_store = None
    if type(params) == NFSDataStoreParameters:
        data_store = NFSDataStore(params)
    elif type(params) == S3DataStoreParameters:
        data_store = S3DataStore(params)

    return data_store

def construct_data_store_params(json: dict):
    ds_params_instance = None
    ds_params = DataStoreParameters(json['store_type'], json['orchestrator_type'], json['orchestrator_params'])
    if json['store_type'] == 'nfs':
        ds_params_instance = NFSDataStoreParameters(ds_params)
    elif json['store_type'] == 's3':
        ds_params_instance = S3DataStoreParameters(ds_params=ds_params,
                                                   end_point=json['end_point'],
                                                   bucket_name=json['bucket_name'],
                                                   checkpoint_dir=json['checkpoint_dir'],
                                                   expt_dir=json['expt_dir'])

    return ds_params_instance
