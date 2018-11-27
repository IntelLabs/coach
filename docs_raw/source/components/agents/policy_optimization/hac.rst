Hierarchical Actor Critic
=========================

**Actions space:** Continuous

**References:** `Hierarchical Reinforcement Learning with Hindsight <https://arxiv.org/abs/1805.08180>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/ddpg.png
   :align: center

Algorithm Description
---------------------
Choosing an action
++++++++++++++++++

Pass the current states through the actor network, and get an action mean vector :math:`\mu`.
While in training phase, use a continuous exploration policy, such as the Ornstein-Uhlenbeck process,
to add exploration noise to the action. When testing, use the mean vector :math:`\mu` as-is.

Training the network
++++++++++++++++++++
