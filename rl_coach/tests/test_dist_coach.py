
from configparser import ConfigParser
import subprocess
import pytest
import argparse

# Test dist coach.

# Test cartpole distributed based on # workers.
# python rl_coach/coach.py  -p CartPole_ClippedPPO -dc -e sample -dcp dist-coach-config.template --dump_worker_logs --checkpoint_save_secs 20

# Test inverted pendulum.
# coach -p Mujoco_ClippedPPO -lvl inverted_pendulum -dc -e sample -dcp dist-coach-config.template --dump_worker_logs --checkpoint_save_secs 20

"""
1. Deploy the examples with a target success rate
2. Wait for the resources to get deleted.
"""

"""
1. Generate the config file and spin up a command.
"""


class Test:

    def __init__(self, command, timeout):
        self.command = command
        self.timeout = timeout


def generate_config(image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file):
    coach_config = ConfigParser({
        'image': image,
        'memory_backend': memory_backend,
        'data_store': data_store,
        's3_end_point': s3_end_point,
        's3_bucket_name': s3_bucket_name,
        's3_creds_file': s3_creds_file
    }, default_section="coach")
    with open(config_file, 'w') as f:
        coach_config.write(f)


def test_command(command, timeout):

    try:
        subprocess.check_call(command, timeout=timeout)
        return 0
    except subprocess.CalledProcessError as e:
        print("{command} did not succeed".format(command=command))
        return 1

    return 1


def test_dc(test, image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file):

    generate_config(image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file)

    command = test.command.format(template=config_file).split(' ')

    assert test_command(command, test.timeout) == 0


def get_tests():
    tests = [
        Test('python rl_coach/coach.py -p CartPole_ClippedPPO -dc -e sample -dcp {template} --dump_worker_logs --checkpoint_save_secs 20 -asc', 200),
        # Test('coach -p Mujoco_ClippedPPO -lvl inverted_pendulum -dc -e sample -dcp {template} --dump_worker_logs --checkpoint_save_secs 20 -asc', 200)
    ]
    return tests


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image', help="(string) Name of the testing image", type=str, default=None
    )
    parser.add_argument(
        '-mb', '--memory_backend', help="(string) Name of the memory backend", type=str, default="redispubsub"
    )
    parser.add_argument(
        '-ds', '--data_store', help="(string) Name of the data store", type=str, default="s3"
    )
    parser.add_argument(
        '-e', '--endpoint', help="(string) Name of the s3 endpoint", type=str, default='s3.amazonaws.com'
    )
    parser.add_argument(
        '-cr', '--creds_file', help="(string) Path of the s3 creds file", type=str, default=None
    )
    parser.add_argument(
        '-b', '--bucket', help="(string) Name of the bucket for s3", type=str, default=None
    )

    args = parser.parse_args()

    if args.data_store == 's3':
        if not args.bucket:
            print("bucket_name required for s3")
            exit(1)
        if not args.creds_file:
            print("creds_file required for s3")
            exit(1)
    for test in get_tests():
        test_dc(test, args.image, args.memory_backend, args.data_store, args.endpoint, args.bucket, args.creds_file, './tmp.cred')


if __name__ == "__main__":
    main()
