Categorical DQN
===============

**Actions space:** Discrete

**References:** `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/distributional_dqn.png
   :align: center

Algorithm Description
---------------------

Training the network
++++++++++++++++++++

1. Sample a batch of transitions from the replay buffer.

2. The Bellman update is projected to the set of atoms representing the :math:`Q` values distribution, such
   that the :math:`i-th` component of the projected update is calculated as follows:

   :math:`(\Phi \hat{T} Z_{\theta}(s_t,a_t))_i=\sum_{j=0}^{N-1}\Big[1-\frac{\lvert[\hat{T}_{z_{j}}]^{V_{MAX}}_{V_{MIN}}-z_i\rvert}{\Delta z}\Big]^1_0 \ p_j(s_{t+1}, \pi(s_{t+1}))`

   where:
   *  :math:`[ \cdot ]` bounds its argument in the range :math:`[a, b]`
   *  :math:`\hat{T}_{z_{j}}` is the Bellman update for atom :math:`z_j`: :math:`\hat{T}_{z_{j}} := r+\gamma z_j`


3. Network is trained with the cross entropy loss between the resulting probability distribution and the target
   probability distribution.   Only the target of the actions that were actually taken is updated.

4. Once in every few thousand steps, weights are copied from the online network to the target network.



.. autoclass:: rl_coach.agents.categorical_dqn_agent.CategoricalDQNAlgorithmParameters
