
from configparser import ConfigParser
import pytest
import argparse
import os
from rl_coach.coach import CoachLauncher
import sys


def generate_config(image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file):

    # Write s3 creds
    aws_config = ConfigParser({
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
    }, default_section='default')
    with open(s3_creds_file, 'w') as f:
        aws_config.write(f)

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


def test_command(command):
    sys.argv = command
    print(sys.argv)
    launcher = CoachLauncher()
    with pytest.raises(SystemExit) as e:
        launcher.launch()
    assert e.value.code == 0


def test_dc(command, image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file):
    generate_config(image, memory_backend, data_store, s3_end_point, s3_bucket_name, s3_creds_file, config_file)
    command = command.format(template=config_file).split(' ')

    test_command(command)


def get_tests():
    tests = [
        'rl_coach/coach.py -p CartPole_ClippedPPO -dc -e sample -dcp {template} --dump_worker_logs --checkpoint_save_secs 20 -asc',
        'rl_coach/coach.py -p Mujoco_ClippedPPO -lvl inverted_pendulum -dc -e sample -dcp {template} --dump_worker_logs --checkpoint_save_secs 20 -asc'
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
        '-cr', '--creds_file', help="(string) Path of the s3 creds file", type=str, default='.aws_creds'
    )
    parser.add_argument(
        '-b', '--bucket', help="(string) Name of the bucket for s3", type=str, default=None
    )

    args = parser.parse_args()

    if args.data_store == 's3':
        if not args.bucket:
            print("bucket_name required for s3")
            exit(1)
        if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
            print("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars need to be set")
            exit(1)
    for command in get_tests():
        test_dc(command, args.image, args.memory_backend, args.data_store, args.endpoint, args.bucket, args.creds_file, './tmp.cred')


if __name__ == "__main__":
    main()
