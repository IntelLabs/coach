# Coach

[![CI](https://img.shields.io/circleci/project/github/NervanaSystems/coach/master.svg)](https://circleci.com/gh/NervanaSystems/workflows/coach/tree/master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/coach/blob/master/LICENSE)
[![Docs](https://readthedocs.org/projects/carla/badge/?version=latest)](https://nervanasystems.github.io/coach/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1134898.svg)](https://doi.org/10.5281/zenodo.1134898)

<p align="center"><img src="img/coach_logo.png" alt="Coach Logo" width="200"/></p>

Coach is a python reinforcement learning framework containing implementation of many state-of-the-art algorithms.

It exposes a set of easy-to-use APIs for experimenting with new RL algorithms, and allows simple integration of new environments to solve. 
Basic RL components (algorithms, environments, neural network architectures, exploration policies, ...) are well decoupled, so that extending and reusing existing components is fairly painless.

Training an agent to solve an environment is as easy as running:

```bash
coach -p CartPole_DQN -r
```

<img src="img/fetch_slide.gif" alt="Fetch Slide"/> <img src="img/pendulum.gif" alt="Pendulum"/> <img src="img/starcraft.gif" width = "281" height ="200" alt="Starcraft"/>
<br>
<img src="img/doom_deathmatch.gif" alt="Doom Deathmatch"/> <img src="img/carla.gif" alt="CARLA"/> <img src="img/montezuma.gif" alt="MontezumaRevenge" width = "164" height ="200"/>
<br>
<img src="img/doom_health.gif" alt="Doom Health Gathering"/> <img src="img/minitaur.gif" alt="PyBullet Minitaur" width = "249" height ="200"/> <img src="img/ant.gif" alt="Gym Extensions Ant"/>
<br><br>

* [Release 0.8.0](https://ai.intel.com/reinforcement-learning-coach-intel/) (initial release)
* [Release 0.9.0](https://ai.intel.com/reinforcement-learning-coach-carla-qr-dqn/)
* [Release 0.10.0](https://ai.intel.com/introducing-reinforcement-learning-coach-0-10-0/)
* [Release 0.11.0](https://ai.intel.com/rl-coach-data-science-at-scale)
* [Release 0.12.0](https://github.com/NervanaSystems/coach/releases/tag/v0.12.0) 
* [Release 1.0.0](https://www.intel.ai/rl-coach-new-release) (current release)


## Table of Contents

- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Getting Started](#getting-started)
  * [Tutorials and Documentation](#tutorials-and-documentation)
  * [Basic Usage](#basic-usage)
    * [Running Coach](#running-coach)
    * [Running Coach Dashboard (Visualization)](#running-coach-dashboard-visualization)
  * [Distributed Multi-Node Coach](#distributed-multi-node-coach)
  * [Batch Reinforcement Learning](#batch-reinforcement-learning)
- [Supported Environments](#supported-environments)
- [Supported Algorithms](#supported-algorithms)
- [Citation](#citation)
- [Contact](#contact)
- [Disclaimer](#disclaimer)

## Benchmarks

One of the main challenges when building a research project, or a solution based on a published algorithm, is getting a concrete and reliable baseline that reproduces the algorithm's results, as reported by its authors. To address this problem, we are releasing a set of [benchmarks](benchmarks) that shows Coach reliably reproduces many state of the art algorithm results.

## Installation

Note: Coach has only been tested on Ubuntu 16.04 LTS, and with Python 3.5.

For some information on installing on Ubuntu 17.10 with Python 3.6.3, please refer to the following issue: https://github.com/NervanaSystems/coach/issues/54

In order to install coach, there are a few prerequisites required. This will setup all the basics needed to get the user going with running Coach on top of [OpenAI Gym](https://github.com/openai/gym) environments:

```
# General
sudo -E apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y

# Boost libraries
sudo -E apt-get install libboost-all-dev -y

# Scipy requirements
sudo -E apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y

# PyGame
sudo -E apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev
libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y

# Dashboard
sudo -E apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev 
freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev
libgstreamer-plugins-base1.0-dev -y

# Gym
sudo -E apt-get install libav-tools libsdl2-dev swig cmake -y
```

We recommend installing coach in a virtualenv:

```
sudo -E pip3 install virtualenv
virtualenv -p python3 coach_env
. coach_env/bin/activate
```

Finally, install coach using pip:
```
pip3 install rl_coach
```

Or alternatively, for a development environment, install coach from the cloned repository:
```
cd coach
pip3 install -e .
```

If a GPU is present, Coach's pip package will install tensorflow-gpu, by default. If a GPU is not present, an [Intel-Optimized TensorFlow](https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available), will be installed. 

In addition to OpenAI Gym, several other environments were tested and are supported. Please follow the instructions in the Supported Environments section below in order to install more environments.

## Getting Started

### Tutorials and Documentation
[Jupyter notebooks demonstrating how to run Coach from command line or as a library, implement an algorithm, or integrate an environment](https://github.com/NervanaSystems/coach/tree/master/tutorials).

[Framework documentation, algorithm description and instructions on how to contribute a new agent/environment](https://nervanasystems.github.io/coach/).

### Basic Usage

#### Running Coach

To allow reproducing results in Coach, we defined a mechanism called _preset_. 
There are several available presets under the `presets` directory.
To list all the available presets use the `-l` flag.

To run a preset, use:

```bash
coach -r -p <preset_name>
```

For example:
* CartPole environment using Policy Gradients (PG):

  ```bash
  coach -r -p CartPole_PG
  ```
  
* Basic level of Doom using Dueling network and Double DQN (DDQN) algorithm:

  ```bash
  coach -r -p Doom_Basic_Dueling_DDQN
  ```

Some presets apply to a group of environment levels, like the entire Atari or Mujoco suites for example.
To use these presets, the requeseted level should be defined using the `-lvl` flag.

For example:


* Pong using the Neural Episodic Control (NEC) algorithm:

  ```bash
  coach -r -p Atari_NEC -lvl pong
  ```

There are several types of agents that can benefit from running them in a distributed fashion with multiple workers in parallel. Each worker interacts with its own copy of the environment but updates a shared network, which improves the data collection speed and the stability of the learning process.
To specify the number of workers to run, use the `-n` flag.

For example:
* Breakout using Asynchronous Advantage Actor-Critic (A3C) with 8 workers:

  ```bash
  coach -r -p Atari_A3C -lvl breakout -n 8
  ```


It is easy to create new presets for different levels or environments by following the same pattern as in presets.py

More usage examples can be found [here](https://github.com/NervanaSystems/coach/blob/master/tutorials/0.%20Quick%20Start%20Guide.ipynb).

#### Running Coach Dashboard (Visualization)
Training an agent to solve an environment can be tricky, at times. 

In order to debug the training process, Coach outputs several signals, per trained algorithm, in order to track algorithmic performance. 

While Coach trains an agent, a csv file containing the relevant training signals will be saved to the 'experiments' directory. Coach's dashboard can then be used to dynamically visualize the training signals, and track algorithmic behavior. 

To use it, run:

```bash
dashboard
```



<img src="img/dashboard.gif" alt="Coach Design" style="width: 800px;"/>


### Distributed Multi-Node Coach

As of release 0.11.0, Coach supports horizontal scaling for training RL agents on multiple nodes. In release 0.11.0 this was tested on the ClippedPPO and DQN agents.
For usage instructions please refer to the documentation [here](https://nervanasystems.github.io/coach/dist_usage.html).

### Batch Reinforcement Learning

Training and evaluating an agent from a dataset of experience, where no simulator is available, is supported in Coach. 
There are [example](https://github.com/NervanaSystems/coach/blob/master/rl_coach/presets/CartPole_DDQN_BatchRL.py) [presets](https://github.com/NervanaSystems/coach/blob/master/rl_coach/presets/Acrobot_DDQN_BCQ_BatchRL.py) and a [tutorial](https://github.com/NervanaSystems/coach/blob/master/tutorials/4.%20Batch%20Reinforcement%20Learning.ipynb). 


## Supported Environments

* *OpenAI Gym:*

    Installed by default by Coach's installer

* *ViZDoom:*

    Follow the instructions described in the ViZDoom repository -

    https://github.com/mwydmuch/ViZDoom

    Additionally, Coach assumes that the environment variable VIZDOOM_ROOT points to the ViZDoom installation directory.

* *Roboschool:*

    Follow the instructions described in the roboschool repository - 

    https://github.com/openai/roboschool

* *GymExtensions:*

    Follow the instructions described in the GymExtensions repository -

    https://github.com/Breakend/gym-extensions

    Additionally, add the installation directory to the PYTHONPATH environment variable.

* *PyBullet:*

    Follow the instructions described in the [Quick Start Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA) (basically just - 'pip install pybullet')

* *CARLA:*

    Download release 0.8.4 from the CARLA repository -

    https://github.com/carla-simulator/carla/releases

    Install the python client and dependencies from the release tarball:
    ```
    pip3 install -r PythonClient/requirements.txt
    pip3 install PythonClient
    ```

    Create a new CARLA_ROOT environment variable pointing to CARLA's installation directory.

    A simple CARLA settings file (```CarlaSettings.ini```) is supplied with Coach, and is located in the ```environments``` directory.

* *Starcraft:*

    Follow the instructions described in the PySC2 repository - 
    
    https://github.com/deepmind/pysc2
    
* *DeepMind Control Suite:*

    Follow the instructions described in the DeepMind Control Suite repository - 
    
    https://github.com/deepmind/dm_control


## Supported Algorithms

<img src="docs_raw/source/_static/img/algorithms.png" alt="Coach Design" style="width: 800px;"/>




### Value Optimization Agents
* [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  ([code](rl_coach/agents/dqn_agent.py))
* [Double Deep Q Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)  ([code](rl_coach/agents/ddqn_agent.py))
* [Dueling Q Network](https://arxiv.org/abs/1511.06581)
* [Mixed Monte Carlo (MMC)](https://arxiv.org/abs/1703.01310)  ([code](rl_coach/agents/mmc_agent.py))
* [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860)  ([code](rl_coach/agents/pal_agent.py))
* [Categorical Deep Q Network (C51)](https://arxiv.org/abs/1707.06887)  ([code](rl_coach/agents/categorical_dqn_agent.py))
* [Quantile Regression Deep Q Network (QR-DQN)](https://arxiv.org/pdf/1710.10044v1.pdf)  ([code](rl_coach/agents/qr_dqn_agent.py))
* [N-Step Q Learning](https://arxiv.org/abs/1602.01783) | **Multi Worker Single Node**  ([code](rl_coach/agents/n_step_q_agent.py))
* [Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)  ([code](rl_coach/agents/nec_agent.py))
* [Normalized Advantage Functions (NAF)](https://arxiv.org/abs/1603.00748.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/naf_agent.py))
* [Rainbow](https://arxiv.org/abs/1710.02298)  ([code](rl_coach/agents/rainbow_dqn_agent.py))

### Policy Optimization Agents
* [Policy Gradients (PG)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/policy_gradients_agent.py))
* [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783) | **Multi Worker Single Node**  ([code](rl_coach/agents/actor_critic_agent.py))
* [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | **Multi Worker Single Node**  ([code](rl_coach/agents/ddpg_agent.py))
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  ([code](rl_coach/agents/ppo_agent.py))
* [Clipped Proximal Policy Optimization (CPPO)](https://arxiv.org/pdf/1707.06347.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/clipped_ppo_agent.py))
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) ([code](rl_coach/agents/actor_critic_agent.py#L86))
* [Sample Efficient Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224) | **Multi Worker Single Node**  ([code](rl_coach/agents/acer_agent.py))
* [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) ([code](rl_coach/agents/soft_actor_critic_agent.py))
* [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf) ([code](rl_coach/agents/td3_agent.py))

### General Agents
* [Direct Future Prediction (DFP)](https://arxiv.org/abs/1611.01779) | **Multi Worker Single Node**  ([code](rl_coach/agents/dfp_agent.py))

### Imitation Learning Agents
* Behavioral Cloning (BC)  ([code](rl_coach/agents/bc_agent.py))
* [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410) ([code](rl_coach/agents/cil_agent.py))

### Hierarchical Reinforcement Learning Agents
* [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf) ([code](rl_coach/agents/hac_ddpg_agent.py))

### Memory Types
* [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf) ([code](rl_coach/memories/episodic/episodic_hindsight_experience_replay.py))
* [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) ([code](rl_coach/memories/non_episodic/prioritized_experience_replay.py))

### Exploration Techniques
* E-Greedy ([code](rl_coach/exploration_policies/e_greedy.py))
* Boltzmann ([code](rl_coach/exploration_policies/boltzmann.py))
* Ornstein–Uhlenbeck process ([code](rl_coach/exploration_policies/ou_process.py))
* Normal Noise ([code](rl_coach/exploration_policies/additive_noise.py))
* Truncated Normal Noise ([code](rl_coach/exploration_policies/truncated_normal.py))
* [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621)  ([code](rl_coach/agents/bootstrapped_dqn_agent.py))
* [UCB Exploration via Q-Ensembles (UCB)](https://arxiv.org/abs/1706.01502) ([code](rl_coach/exploration_policies/ucb.py))
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) ([code](rl_coach/exploration_policies/parameter_noise.py))

## Citation

If you used Coach for your work, please use the following citation:

```
@misc{caspi_itai_2017_1134899,
  author       = {Caspi, Itai and
                  Leibovich, Gal and
                  Novik, Gal and
                  Endrawis, Shadi},
  title        = {Reinforcement Learning Coach},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1134899},
  url          = {https://doi.org/10.5281/zenodo.1134899}
}
```

## Contact

We'd be happy to get any questions or contributions through GitHub issues and PRs.

Please make sure to take a look [here](CONTRIBUTING.md) before filing an issue or proposing a PR.

The Coach development team can also be contacted over [email](mailto:coach@intel.com)


## Disclaimer

Coach is released as a reference code for research purposes. It is not an official Intel product, and the level of quality and support may not be as expected from an official product. 
Additional algorithms and environments are planned to be added to the framework. Feedback and contributions from the open source and RL research communities are more than welcome.
