Quantile Regression DQN
=======================

**Actions space:** Discrete

**References:** `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/abs/1710.10044>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/qr_dqn.png
   :align: center

Algorithm Description
---------------------

Training the network
++++++++++++++++++++

1. Sample a batch of transitions from the replay buffer.

2. First, the next state quantiles are predicted. These are used in order to calculate the targets for the network,
   by following the Bellman equation.
   Next, the current quantile locations for the current states are predicted, sorted, and used for calculating the
   quantile midpoints targets.

3. The network is trained with the quantile regression loss between the resulting quantile locations and the target
   quantile locations. Only the targets of the actions that were actually taken are updated.

4. Once in every few thousand steps, weights are copied from the online network to the target network.


.. autoclass:: rl_coach.agents.qr_dqn_agent.QuantileRegressionDQNAlgorithmParameters