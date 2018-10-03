import argparse

from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes, RunTypeParameters
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters


def main(preset: str, image: str='ajaysudh/testing:coach', num_workers: int=1, nfs_server: str="", nfs_path: str="", memory_backend: str=""):
    rollout_command = ['python3', 'rl_coach/rollout_worker.py', '-p', preset]
    training_command = ['python3', 'rl_coach/training_worker.py', '-p', preset]

    memory_backend_params = RedisPubSubMemoryBackendParameters()

    worker_run_type_params = RunTypeParameters(image, rollout_command, run_type="worker")
    trainer_run_type_params = RunTypeParameters(image, training_command, run_type="trainer")

    orchestration_params = KubernetesParameters([worker_run_type_params, trainer_run_type_params], kubeconfig='~/.kube/config', nfs_server=nfs_server,
                                                nfs_path=nfs_path, memory_backend_parameters=memory_backend_params)
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
                        help="(string) Name of a preset to run (class name from the 'presets' directory.)",
                        type=str,
                        required=True)
    parser.add_argument('-ns', '--nfs-server',
                        help="(string) Addresss of the nfs server.)",
                        type=str,
                        required=True)
    parser.add_argument('-np', '--nfs-path',
                        help="(string) Exported path for the nfs server",
                        type=str,
                        required=True)
    parser.add_argument('--memory_backend',
                        help="(string) Memory backend to use",
                        type=str,
                        default="redispubsub")

    # parser.add_argument('--checkpoint_dir',
    #                     help='(string) Path to a folder containing a checkpoint to write the model to.',
    #                     type=str,
    #                     default='/checkpoint')
    args = parser.parse_args()

    main(preset=args.preset, image=args.image, nfs_server=args.nfs_server, nfs_path=args.nfs_path, memory_backend=args.memory_backend)
