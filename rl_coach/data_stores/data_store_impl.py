from rl_coach.data_stores.nfs_data_store import NFSDataStore, NFSDataStoreParameters
from rl_coach.data_stores.s3_data_store import S3DataStore, S3DataStoreParameters


def get_data_store(params):
    data_store = None
    if type(params) == NFSDataStoreParameters:
        data_store = NFSDataStore(params)
    elif type(params) == S3DataStoreParameters:
        data_store = S3DataStore(params)

    return data_store
