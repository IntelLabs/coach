Usage
=====

One of the mechanisms Coach uses for running experiments is the **Preset** mechanism.
As its name implies, a preset defines a set of predefined experiment parameters.
This allows defining a *complex* agent-environment interaction, with multiple parameters, and later running it through
a very *simple* command line.

The preset includes all the components that are used in the experiment, such as the agent internal components and
the environment to use.
It additionally defines general parameters for the experiment itself, such as the training schedule,
visualization parameters, and testing parameters.

Training an Agent
-----------------

Single-threaded Algorithms
++++++++++++++++++++++++++

This is the most common case. Just choose a preset using the `-p` flag and press enter.
To list the available presets, use the `-l` flag.

*Example:*

.. code-block:: python

   coach -p CartPole_DQN

Multi-threaded Algorithms
+++++++++++++++++++++++++

Multi-threaded algorithms are very common these days.
They typically achieve the best results, and scale gracefully with the number of threads.
In Coach, running such algorithms is done by selecting a suitable preset, and choosing the number of threads to run using the :code:`-n` flag.

*Example:*

.. code-block:: python

   coach -p CartPole_A3C -n 8

Multi-Node Algorithms
+++++++++++++++++++++++++

Coach supports the multi-node runs in distributed mode. Specifically, the horizontal scale-out of rollout workers is implemented.
In Coach, running such algorithms is done by selecting a suitable preset, enabling distributed coach using :code:`-dc` flag,
passing distributed coach parameters using :code:`dcp` and choosing the number of to run using the :code:`-n` flag.
For more details and instructions on how to use distributed Coach, see :ref:`dist-coach-usage`.

*Example:*

.. code-block:: python

   coach -p CartPole_ClippedPPO -dc -dcp <path-to-config-file> -n 8

Evaluating an Agent
-------------------

There are several options for evaluating an agent during the training:

* For multi-threaded runs, an evaluation agent will constantly run in the background and evaluate the model during the training.

* For single-threaded runs, it is possible to define an evaluation period through the preset. This will run several episodes of evaluation once in a while.

Additionally, it is possible to save checkpoints of the agents networks and then run only in evaluation mode.
Saving checkpoints can be done by specifying the number of seconds between storing checkpoints using the :code:`-s` flag.
The checkpoints will be saved into the experiment directory.
Loading a model for evaluation can be done by specifying the :code:`-crd` flag with the experiment directory, and the :code:`--evaluate` flag to disable training.

*Example:*

.. code-block:: python

   coach -p CartPole_DQN -s 60
   coach -p CartPole_DQN --evaluate -crd CHECKPOINT_RESTORE_DIR

Playing with the Environment as a Human
---------------------------------------

Interacting with the environment as a human can be useful for understanding its difficulties and for collecting data for imitation learning.
In Coach, this can be easily done by selecting a preset that defines the environment to use, and specifying the :code:`--play` flag.
When the environment is loaded, the available keyboard buttons will be printed to the screen.
Pressing the escape key when finished will end the simulation and store the replay buffer in the experiment dir.

*Example:*

.. code-block:: python

   coach -et rl_coach.environments.gym_environment:Atari -lvl BreakoutDeterministic-v4 --play

Learning Through Imitation Learning
-----------------------------------

Learning through imitation of human behavior is a nice way to speedup the learning.
In Coach, this can be done in two steps -

1. Create a dataset of demonstrations by playing with the environment as a human.
   After this step, a pickle of the replay buffer containing your game play will be stored in the experiment directory.
   The path to this replay buffer will be printed to the screen.
   To do so, you should select an environment type and level through the command line, and specify the :code:`--play` flag.

    *Example:*

.. code-block:: python

   coach -et rl_coach.environments.doom_environment:DoomEnvironmentParameters -lvl Basic --play


2. Next, use an imitation learning preset and set the replay buffer path accordingly.
    The path can be set either from the command line or from the preset itself.

    *Example:*

.. code-block:: python

    from rl_coach.core_types import PickledReplayBuffer
    coach -p Doom_Basic_BC -cp='agent.load_memory_from_file_path=PickledReplayBuffer(\"<experiment dir>/replay_buffer.p\"')


Visualizations
--------------

Rendering the Environment
+++++++++++++++++++++++++

Rendering the environment can be done by using the :code:`-r` flag.
When working with multi-threaded algorithms, the rendered image will be representing the game play of the evaluation worker.
When working with single-threaded algorithms, the rendered image will be representing the single worker which can be either training or evaluating.
Keep in mind that rendering the environment in single-threaded algorithms may slow the training to some extent.
When playing with the environment using the :code:`--play` flag, the environment will be rendered automatically without the need for specifying the :code:`-r` flag.

*Example:*

.. code-block:: python

   coach -p Atari_DQN -lvl breakout -r

Dumping GIFs
++++++++++++

Coach allows storing GIFs of the agent game play.
To dump GIF files, use the :code:`-dg` flag.
The files are dumped after every evaluation episode, and are saved into the experiment directory, under a gifs sub-directory.

*Example:*

.. code-block:: python

   coach -p Atari_A3C -lvl breakout -n 4 -dg

Switching Between Deep Learning Frameworks
------------------------------------------

Coach uses TensorFlow as its main backend framework, but it also supports MXNet.
MXNet is optional, and by default, TensorFlow will be used.
If MXNet was installed, it is possible to switch to MXNet using the :code:`-f` flag.

*Example:*

.. code-block:: python

   coach -p Doom_Basic_DQN -f mxnet

Additional Flags
----------------

There are several convenient flags which are important to know about.
The most up to date description can be found by using the :code:`-h` flag.

.. argparse::
   :module: rl_coach.coach
   :func: create_argument_parser
   :prog: coach
