import argparse

from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes, RunTypeParameters
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.data_stores.data_store import DataStoreParameters
from rl_coach.data_stores.s3_data_store import S3DataStoreParameters
from rl_coach.data_stores.nfs_data_store import NFSDataStoreParameters


def main(preset: str, image: str='ajaysudh/testing:coach', num_workers: int=1, nfs_server: str=None, nfs_path: str=None,
         memory_backend: str=None, data_store: str=None, s3_end_point: str=None, s3_bucket_name: str=None):
    rollout_command = ['python3', 'rl_coach/rollout_worker.py', '-p', preset]
    training_command = ['python3', 'rl_coach/training_worker.py', '-p', preset]

    memory_backend_params = None
    if memory_backend == "redispubsub":
        memory_backend_params = RedisPubSubMemoryBackendParameters()

    ds_params_instance = None
    if data_store == "s3":
        ds_params = DataStoreParameters("s3", "", "")
        ds_params_instance = S3DataStoreParameters(ds_params=ds_params, end_point=s3_end_point, bucket_name=s3_bucket_name,
                                                   checkpoint_dir="/checkpoint")
    elif data_store == "nfs":
        ds_params = DataStoreParameters("nfs", "kubernetes", {"namespace": "default"})
        ds_params_instance = NFSDataStoreParameters(ds_params)

    worker_run_type_params = RunTypeParameters(image, rollout_command, run_type="worker", num_replicas=num_workers)
    trainer_run_type_params = RunTypeParameters(image, training_command, run_type="trainer")

    orchestration_params = KubernetesParameters([worker_run_type_params, trainer_run_type_params],
                                                kubeconfig='~/.kube/config', nfs_server=nfs_server, nfs_path=nfs_path,
                                                memory_backend_parameters=memory_backend_params,
                                                data_store_params=ds_params_instance)
    orchestrator = Kubernetes(orchestration_params)
    if not orchestrator.setup():
        print("Could not setup")
        return

    if orchestrator.deploy_trainer():
        print("Successfully deployed")
    else:
        print("Could not deploy")
        return

    if orchestrator.deploy_worker():
        print("Successfully deployed")
    else:
        print("Could not deploy")
        return

    try:
        orchestrator.trainer_logs()
    except KeyboardInterrupt:
        pass
    orchestrator.undeploy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        help="(string) Name of a docker image.",
                        type=str,
                        required=True)
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run (class name from the 'presets' directory).",
                        type=str,
                        required=True)
    parser.add_argument('--memory-backend',
                        help="(string) Memory backend to use.",
                        type=str,
                        default="redispubsub")
    parser.add_argument('-ds', '--data-store',
                        help="(string) Data store to use.",
                        type=str,
                        default="s3")
    parser.add_argument('-ns', '--nfs-server',
                        help="(string) Addresss of the nfs server.",
                        type=str,
                        required=True)
    parser.add_argument('-np', '--nfs-path',
                        help="(string) Exported path for the nfs server.",
                        type=str,
                        required=True)
    parser.add_argument('--s3-end-point',
                        help="(string) S3 endpoint to use when S3 data store is used.",
                        type=str,
                        required=True)
    parser.add_argument('--s3-bucket-name',
                        help="(string) S3 bucket name to use when S3 data store is used.",
                        type=str,
                        required=True)
    parser.add_argument('--num-workers',
                        help="(string) Number of rollout workers",
                        type=int,
                        required=False,
                        default=1)

    # parser.add_argument('--checkpoint_dir',
    #                     help='(string) Path to a folder containing a checkpoint to write the model to.',
    #                     type=str,
    #                     default='/checkpoint')
    args = parser.parse_args()

    main(preset=args.preset, image=args.image, nfs_server=args.nfs_server, nfs_path=args.nfs_path,
         memory_backend=args.memory_backend, data_store=args.data_store, s3_end_point=args.s3_end_point,
         s3_bucket_name=args.s3_bucket_name, num_workers=args.num_workers)
