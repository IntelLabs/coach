# Coach Features

## Supported Algorithms

Coach supports many state-of-the-art reinforcement learning algorithms, which are separated into two main classes -
value optimization and policy optimization. A detailed description of those algorithms may be found in the algorithms
section.

<p style="text-align: center;">

<img src="../../img/algorithms.png" alt="Supported Algorithms" style="width: 600px;"/>

</p>


## Supported Environments

Coach supports a large number of environments which can be solved using reinforcement learning:

* **[DeepMind Control Suite](https://github.com/deepmind/dm_control)** - a set of reinforcement learning environments
  powered by the MuJoCo physics engine.

* **[Blizzard Starcraft II](https://github.com/deepmind/pysc2)** - a popular strategy game which was wrapped with a
  python interface by DeepMind.

* **[ViZDoom](http://vizdoom.cs.put.edu.pl/)** - a Doom-based AI research platform for reinforcement learning
  from raw visual information.

* **[CARLA](https://github.com/carla-simulator/carla)** - an open-source simulator for autonomous driving research.

* **[OpenAI Gym](https://gym.openai.com/)** - a library which consists of a set of environments, from games to robotics.
  Additionally, it can be extended using the API defined by the authors.

  In Coach, we support all the native environments in Gym, along with several extensions such as:

* **[Roboschool](https://github.com/openai/roboschool)** - a set of environments powered by the PyBullet engine,
    that offer a free alternative to MuJoCo.

* **[Gym Extensions](https://github.com/Breakend/gym-extensions)** - a set of environments that extends Gym for
    auxiliary tasks (multitask learning, transfer learning, inverse reinforcement learning, etc.)

* **[PyBullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)** - a physics engine that
    includes a set of robotics environments.

