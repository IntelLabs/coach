Environments
============

.. autoclass:: rl_coach.environments.environment.Environment
   :members:
   :inherited-members:

DeepMind Control Suite
----------------------

A set of reinforcement learning environments powered by the MuJoCo physics engine.

Website: `DeepMind Control Suite <https://github.com/deepmind/dm_control>`_

.. autoclass:: rl_coach.environments.control_suite_environment.ControlSuiteEnvironment


Blizzard Starcraft II
---------------------

A popular strategy game which was wrapped with a python interface by DeepMind.

Website: `Blizzard Starcraft II <https://github.com/deepmind/pysc2>`_

.. autoclass:: rl_coach.environments.starcraft2_environment.StarCraft2Environment


ViZDoom
--------

A Doom-based AI research platform for reinforcement learning from raw visual information.

Website: `ViZDoom <http://vizdoom.cs.put.edu.pl/>`_

.. autoclass:: rl_coach.environments.doom_environment.DoomEnvironment


CARLA
-----

An open-source simulator for autonomous driving research.

Website: `CARLA <https://github.com/carla-simulator/carla>`_

.. autoclass:: rl_coach.environments.carla_environment.CarlaEnvironment

OpenAI Gym
----------

A library which consists of a set of environments, from games to robotics.
Additionally, it can be extended using the API defined by the authors.

Website: `OpenAI Gym <https://gym.openai.com/>`_

In Coach, we support all the native environments in Gym, along with several extensions such as:

* `Roboschool <https://github.com/openai/roboschool>`_  - a set of environments powered by the PyBullet engine,
  that offer a free alternative to MuJoCo.

* `Gym Extensions <https://github.com/Breakend/gym-extensions>`_  - a set of environments that extends Gym for
  auxiliary tasks (multitask learning, transfer learning, inverse reinforcement learning, etc.)

* `PyBullet <https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet>`_  - a physics engine that
  includes a set of robotics environments.


.. autoclass:: rl_coach.environments.gym_environment.GymEnvironment



