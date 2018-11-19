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

import sys

from codecs import open
from os import path

from setuptools import setup, find_packages
import subprocess

# Creating the pip package involves the following steps:
# - Define the pip package related files - setup.py (this file) and MANIFEST.in by:
# 1. Make sure all the requirements in install_requires are defined correctly and that their version is the correct one
# 2. Add all the non .py files to the package_data and to the MANIFEST.in file
# 3. Make sure that all the python directories have an __init__.py file

# - Check that everything works fine by:
# 1. Create a new virtual environment using `virtualenv coach_env -p python3`
# 2. Run `pip install -e .`
# 3. Run `coach -p CartPole_DQN` and make sure it works
# 4. Run `dashboard` and make sure it works

# - If everything works fine, build and upload the package to PyPi:
# 1. Update the version of Coach in the call to setup()
# 2. Remove the directories build, dist and rl_coach.egg-info if they exist
# 3. Run `python setup.py sdist`
# 4. Run `twine upload dist/*`

slim_package = False  # if true build aws package with partial dependencies, otherwise, build full package

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = list()
extras = dict()
excluded_packages = ['wxPython', 'kubernetes', 'tensorflow'] if slim_package else []

with open(path.join(here, 'requirements.txt'), 'r') as f:
    for line in f:
        package = line.strip()
        if any(p in package for p in excluded_packages):
            continue
        install_requires.append(package)

# check if system has CUDA enabled GPU
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('UTF-8')
using_GPU = out != ''

if not using_GPU:
    if not slim_package:
        # For linux wth no GPU, we install the Intel optimized version of TensorFlow
        if sys.platform == "linux" or sys.platform == "linux2":
            subprocess.check_call(['pip install '
                                   'https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp35-cp35m-linux_x86_64.whl'],
                                  shell=True)
        install_requires.append('tensorflow>=1.9.0')
    extras['mxnet'] = ['mxnet-mkl>=1.3.0']
else:
    if not slim_package:
        install_requires.append('tensorflow-gpu>=1.9.0')
    extras['mxnet'] = ['mxnet-cu90mkl>=1.3.0']

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps


setup(
    name='rl-coach' if not slim_package else 'rl-coach-slim',
    version='0.11.0',
    description='Reinforcement Learning Coach enables easy experimentation with state of the art Reinforcement Learning algorithms.',
    url='https://github.com/NervanaSystems/coach',
    author='Intel AI Lab',
    author_email='coach@intel.com',
    packages=find_packages(),
    python_requires=">=3.5.*",
    install_requires=install_requires,
    extras_require=extras,
    package_data={'rl_coach': ['dashboard_components/*.css',
                               'environments/doom/*.cfg',
                               'environments/doom/*.wad',
                               'environments/mujoco/common/*.xml',
                               'environments/mujoco/*.xml',
                               'environments/*.ini',
                               'tests/*.ini']},
    entry_points={
        'console_scripts': [
            'coach=rl_coach.coach:main',
            'dashboard=rl_coach.dashboard:main'
        ],
    }
)
