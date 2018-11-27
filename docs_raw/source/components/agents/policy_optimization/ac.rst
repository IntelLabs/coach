Actor-Critic
============

**Actions space:** Discrete | Continuous

**References:** `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/ac.png
   :width: 500px
   :align: center

Algorithm Description
---------------------

Choosing an action - Discrete actions
+++++++++++++++++++++++++++++++++++++

The policy network is used in order to predict action probabilites. While training, a sample is taken from a categorical
distribution assigned with these probabilities. When testing, the action with the highest probability is used.

Training the network
++++++++++++++++++++
A batch of :math:`T_{max}` transitions is used, and the advantages are calculated upon it.

Advantages can be calculated by either of the following methods (configured by the selected preset) -

1. **A_VALUE** - Estimating advantage directly:
   :math:`A(s_t, a_t) = \underbrace{\sum_{i=t}^{i=t + k - 1} \gamma^{i-t}r_i +\gamma^{k} V(s_{t+k})}_{Q(s_t, a_t)} - V(s_t)`
   where :math:`k` is :math:`T_{max} - State\_Index` for each state in the batch.

2. **GAE** - By following the `Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_ paper.

The advantages are then used in order to accumulate gradients according to 
:math:`L = -\mathop{\mathbb{E}} [log (\pi) \cdot A]`


.. autoclass:: rl_coach.agents.actor_critic_agent.ActorCriticAlgorithmParameters