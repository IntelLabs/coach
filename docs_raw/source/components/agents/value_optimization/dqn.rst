Deep Q Networks
===============

**Actions space:** Discrete

**References:** `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/dqn.png
   :align: center

Algorithm Description
---------------------

Training the network
++++++++++++++++++++

1. Sample a batch of transitions from the replay buffer.

2. Using the next states from the sampled batch, run the target network to calculate the :math:`Q` values for each of
   the actions :math:`Q(s_{t+1},a)`, and keep only the maximum value for each state.

3. In order to zero out the updates for the actions that were not played (resulting from zeroing the MSE loss),
   use the current states from the sampled batch, and run the online network to get the current Q values predictions.
   Set those values as the targets for the actions that were not actually played.

4. For each action that was played, use the following equation for calculating the targets of the network:​                                                         $$ y_t=r(s_t,a_t)+γ\cdot max_a {Q(s_{t+1},a)} $$ 
   :math:`y_t=r(s_t,a_t )+\gamma \cdot max_a Q(s_{t+1})`

5. Finally, train the online network using the current states as inputs, and with the aforementioned targets.

6. Once in every few thousand steps, copy the weights from the online network to the target network.


.. autoclass:: rl_coach.agents.dqn_agent.DQNAlgorithmParameters
