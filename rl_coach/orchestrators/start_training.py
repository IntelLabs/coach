import argparse

from rl_coach.orchestrators.kubernetes_orchestrator import KubernetesParameters, Kubernetes


def main(preset, image='ajaysudh/testing:coach', redis_ip='redis-service.ajay.svc'):
    rollout_command = ['python3', 'rl_coach/rollout_worker.py', '-p', preset]
    training_command = ['python3', 'rl_coach/training_worker.py', '-p', preset]

    rollout_params = KubernetesParameters(image, rollout_command, redis_ip=redis_ip, redis_port=6379, num_workers=1)
    training_params = KubernetesParameters(image, training_command, redis_ip=redis_ip, redis_port=6379, num_workers=1)

    training_obj = Kubernetes(training_params)
    if not training_obj.setup():
        print("Could not setup")

    rollout_obj = Kubernetes(training_params)
    if not rollout_obj.setup():
        print("Could not setup")

    if training_obj.deploy():
        print("Successfully deployed")
    else:
        print("Could not deploy")

    if rollout_obj.deploy():
        print("Successfully deployed")
    else:
        print("Could not deploy")


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
    # parser.add_argument('--checkpoint_dir',
    #                     help='(string) Path to a folder containing a checkpoint to write the model to.',
    #                     type=str,
    #                     default='/checkpoint')
    args = parser.parse_args()

    main(preset=args.preset, image=args.image)
