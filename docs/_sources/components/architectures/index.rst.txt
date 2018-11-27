Architectures
=============

Architectures contain all the classes that implement the neural network related stuff for the agent.
Since Coach is intended to work with multiple neural network frameworks, each framework will implement its
own components under a dedicated directory. For example, tensorflow components will contain all the neural network
parts that are implemented using TensorFlow.

.. autoclass:: rl_coach.base_parameters.NetworkParameters

Architecture
------------
.. autoclass:: rl_coach.architectures.architecture.Architecture
   :members:
   :inherited-members:

NetworkWrapper
--------------

.. image:: /_static/img/distributed.png
   :width: 600px
   :align: center

.. autoclass:: rl_coach.architectures.network_wrapper.NetworkWrapper
   :members:
   :inherited-members:

