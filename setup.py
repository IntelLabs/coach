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


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires=[
        'annoy==1.8.3', 'Pillow==4.3.0', 'matplotlib==2.0.2', 'numpy==1.14.5', 'pandas==0.22.0',
        'pygame==1.9.3', 'PyOpenGL==3.1.0', 'scipy==0.19.0', 'scikit-image==0.13.0',
        'box2d==2.3.2', 'gym==0.10.5', 'bokeh==0.13.0', 'futures==3.1.1', 'wxPython==4.0.1']

# check if system has CUDA enabled GPU
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('UTF-8')
using_GPU = out != ''

if not using_GPU:
    # For linux wth no GPU, we install the Intel optimized version of TensorFlow
    if sys.platform == "linux" or sys.platform == "linux2":
        subprocess.check_call(['pip install '
                               'https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl'],
                              shell=True)
    install_requires.append('tensorflow==1.6.0')
else:
    install_requires.append('tensorflow-gpu==1.9.0')

setup(
    name='rl-coach',
    version='0.10.0',
    description='Reinforcement Learning Coach enables easy experimentation with state of the art Reinforcement Learning algorithms.',
    url='https://github.com/NervanaSystems/coach',
    author='Intel AI Lab',
    author_email='coach@intel.com',
    packages=find_packages(),
    python_requires=">=3.5.*",
    install_requires=install_requires,
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
