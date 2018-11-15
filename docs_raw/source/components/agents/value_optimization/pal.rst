Persistent Advantage Learning
=============================

**Actions space:** Discrete

**References:** `Increasing the Action Gap: New Operators for Reinforcement Learning <https://arxiv.org/abs/1512.04860>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/dqn.png
   :align: center

Algorithm Description
---------------------
Training the network
++++++++++++++++++++

1. Sample a batch of transitions from the replay buffer. 

2. Start by calculating the initial target values in the same manner as they are calculated in DDQN
   :math:`y_t^{DDQN}=r(s_t,a_t )+\gamma Q(s_{t+1},argmax_a Q(s_{t+1},a))`

3. The action gap :math:`V(s_t )-Q(s_t,a_t)` should then be subtracted from each of the calculated targets.
   To calculate the action gap, run the target network using the current states and get the :math:`Q` values
   for all the actions. Then estimate :math:`V` as the maximum predicted :math:`Q` value for the current state:
   :math:`V(s_t )=max_a Q(s_t,a)`

4. For *advantage learning (AL)*, reduce the action gap weighted by a predefined parameter :math:`\alpha` from
   the targets :math:`y_t^{DDQN}`:
   :math:`y_t=y_t^{DDQN}-\alpha \cdot (V(s_t )-Q(s_t,a_t ))`

5. For *persistent advantage learning (PAL)*, the target network is also used in order to calculate the action
   gap for the next state:
   :math:`V(s_{t+1} )-Q(s_{t+1},a_{t+1})`
   where :math:`a_{t+1}` is chosen by running the next states through the online network and choosing the action that
   has the highest predicted :math:`Q` value. Finally, the targets will be defined as -
   :math:`y_t=y_t^{DDQN}-\alpha \cdot min(V(s_t )-Q(s_t,a_t ),V(s_{t+1} )-Q(s_{t+1},a_{t+1} ))`

6. Train the online network using the current states as inputs, and with the aforementioned targets.

7. Once in every few thousand steps, copy the weights from the online network to the target network.


.. autoclass:: rl_coach.agents.pal_agent.PALAlgorithmParameters
