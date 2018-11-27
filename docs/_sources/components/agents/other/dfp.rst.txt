Direct Future Prediction
========================

**Actions space:** Discrete

**References:** `Learning to Act by Predicting the Future <https://arxiv.org/abs/1611.01779>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/dfp.png
   :width: 600px
   :align: center


Algorithm Description
---------------------
Choosing an action
++++++++++++++++++

1. The current states (observations and measurements) and the corresponding goal vector are passed as an input to the network.
   The output of the network is the predicted future measurements for time-steps :math:`t+1,t+2,t+4,t+8,t+16` and
   :math:`t+32` for each possible action.
2. For each action, the measurements of each predicted time-step are multiplied by the goal vector,
   and the result is a single vector of future values for each action.
3. Then, a weighted sum of the future values of each action is calculated, and the result is a single value for each action. 
4. The action values are passed to the exploration policy to decide on the action to use.

Training the network
++++++++++++++++++++

Given a batch of transitions, run them through the network to get the current predictions of the future measurements
per action, and set them as the initial targets for training the network. For each transition
:math:`(s_t,a_t,r_t,s_{t+1} )` in the batch, the target of the network for the action that was taken, is the actual
 measurements that were seen in time-steps :math:`t+1,t+2,t+4,t+8,t+16` and :math:`t+32`.
 For the actions that were not taken, the targets are the current values.


.. autoclass:: rl_coach.agents.dfp_agent.DFPAlgorithmParameters
