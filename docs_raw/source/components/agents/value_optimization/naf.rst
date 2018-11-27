Normalized Advantage Functions
==============================

**Actions space:** Continuous

**References:** `Continuous Deep Q-Learning with Model-based Acceleration <https://arxiv.org/abs/1603.00748.pdf>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/naf.png
   :width: 600px
   :align: center

Algorithm Description
---------------------
Choosing an action
++++++++++++++++++
The current state is used as an input to the network. The action mean :math:`\mu(s_t )` is extracted from the output head.
It is then passed to the exploration policy which adds noise in order to encourage exploration.

Training the network
++++++++++++++++++++
The network is trained by using the following targets:
:math:`y_t=r(s_t,a_t )+\gamma\cdot V(s_{t+1})`
Use the next states as the inputs to the target network and extract the :math:`V` value, from within the head,
to get :math:`V(s_{t+1} )`. Then, update the online network using the current states and actions as inputs,
and :math:`y_t` as the targets.
After every training step, use a soft update in order to copy the weights from the online network to the target network.



.. autoclass:: rl_coach.agents.naf_agent.NAFAlgorithmParameters
