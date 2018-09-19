import argparse

from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes


def main(preset: str, image: str='ajaysudh/testing:coach', redis_ip: str=None, redis_port:int=None, num_workers: int=1, nfs_server: str="", nfs_path: str=""):
    rollout_command = ['python3', 'rl_coach/rollout_worker.py', '-p', preset]
    training_command = ['python3', 'rl_coach/training_worker.py', '-p', preset]

    """
    TODO:
    1. Create a NFS backed PV for checkpointing.
        a. Include that in both (worker, trainer) containers.
        b. Change checkpoint writing logic to always write to a temporary file and then rename.
    2. Test e2e 1 loop.
        a. Trainer writes a checkpoint
        b. Rollout worker picks it and gathers experience, writes back to redis.
        c. 1 rollout worker, 1 trainer.
    3. Trainer should be a job (not a deployment)
        a. When all the epochs of training are done, workers should also be deleted.
    4. Test e2e with multiple rollout workers.
    5. Test e2e with multiple rollout workers and multiple loops.
    """

    training_params = KubernetesParameters("train", image, training_command, kubeconfig='~/.kube/config', redis_ip=redis_ip, redis_port=redis_port,
                                           nfs_server=nfs_server, nfs_path=nfs_path)
    training_obj = Kubernetes(training_params)
    if not training_obj.setup():
        print("Could not setup")
        return

    rollout_params = KubernetesParameters("worker", image, rollout_command, kubeconfig='~/.kube/config', redis_ip=training_params.redis_ip, redis_port=training_params.redis_port, num_workers=num_workers)
    rollout_obj = Kubernetes(rollout_params)
    # if not rollout_obj.setup():
    #     print("Could not setup")

    if training_obj.deploy():
        print("Successfully deployed")
    else:
        print("Could not deploy")
        return

    if rollout_obj.deploy():
        print("Successfully deployed")
    else:
        print("Could not deploy")
        return


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

    # parser.add_argument('--checkpoint_dir',
    #                     help='(string) Path to a folder containing a checkpoint to write the model to.',
    #                     type=str,
    #                     default='/checkpoint')
    args = parser.parse_args()

    main(preset=args.preset, image=args.image, nfs_server=args.nfs_server, nfs_path=args.nfs_path)
