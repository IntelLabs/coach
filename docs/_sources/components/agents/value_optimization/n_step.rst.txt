N-Step Q Learning
=================

**Actions space:** Discrete

**References:** `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/dqn.png
   :align: center

Algorithm Description
---------------------

Training the network
++++++++++++++++++++

The :math:`N`-step Q learning algorithm works in similar manner to DQN except for the following changes:

1. No replay buffer is used. Instead of sampling random batches of transitions, the network is trained every
   :math:`N` steps using the latest :math:`N` steps played by the agent.

2. In order to stabilize the learning, multiple workers work together to update the network.
   This creates the same effect as uncorrelating the samples used for training.

3. Instead of using single-step Q targets for the network, the rewards from $N$ consequent steps are accumulated
   to form the :math:`N`-step Q targets, according to the following equation:
   :math:`R(s_t, a_t) = \sum_{i=t}^{i=t + k - 1} \gamma^{i-t}r_i +\gamma^{k} V(s_{t+k})`
   where :math:`k` is :math:`T_{max} - State\_Index` for each state in the batch



.. autoclass:: rl_coach.agents.n_step_q_agent.NStepQAlgorithmParameters
