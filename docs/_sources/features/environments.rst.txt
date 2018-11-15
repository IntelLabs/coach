Environments
============

Coach supports a large number of environments which can be solved using reinforcement learning.
To find a detailed documentation of the environments API, see the `environments section <../components/environments/index.html>`_.
The supported environments are:

* `DeepMind Control Suite <https://github.com/deepmind/dm_control>`_  - a set of reinforcement learning environments
  powered by the MuJoCo physics engine.

* `Blizzard Starcraft II <https://github.com/deepmind/pysc2>`_  - a popular strategy game which was wrapped with a
  python interface by DeepMind.

* `ViZDoom <http://vizdoom.cs.put.edu.pl/>`_  - a Doom-based AI research platform for reinforcement learning
  from raw visual information.

* `CARLA <https://github.com/carla-simulator/carla>`_  - an open-source simulator for autonomous driving research.

* `OpenAI Gym <https://gym.openai.com/>`_  - a library which consists of a set of environments, from games to robotics.
  Additionally, it can be extended using the API defined by the authors.

  In Coach, we support all the native environments in Gym, along with several extensions such as:

  * `Roboschool <https://github.com/openai/roboschool>`_  - a set of environments powered by the PyBullet engine,
    that offer a free alternative to MuJoCo.

  * `Gym Extensions <https://github.com/Breakend/gym-extensions>`_  - a set of environments that extends Gym for
    auxiliary tasks (multitask learning, transfer learning, inverse reinforcement learning, etc.)

  * `PyBullet <https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet>`_  - a physics engine that
    includes a set of robotics environments.
