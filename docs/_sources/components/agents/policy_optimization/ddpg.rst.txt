Deep Deterministic Policy Gradient
==================================

**Actions space:** Continuous

**References:** `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_

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

Start by sampling a batch of transitions from the experience replay.

* To train the **critic network**, use the following targets:

  :math:`y_t=r(s_t,a_t )+\gamma \cdot Q(s_{t+1},\mu(s_{t+1} ))`

  First run the actor target network, using the next states as the inputs, and get :math:`\mu (s_{t+1} )`.
  Next, run the critic target network using the next states and :math:`\mu (s_{t+1} )`, and use the output to
  calculate :math:`y_t` according to the equation above. To train the network, use the current states and actions
  as the inputs, and :math:`y_t` as the targets.

* To train the **actor network**, use the following equation:

  :math:`\nabla_{\theta^\mu } J \approx E_{s_t \tilde{} \rho^\beta } [\nabla_a Q(s,a)|_{s=s_t,a=\mu (s_t ) } \cdot \nabla_{\theta^\mu} \mu(s)|_{s=s_t} ]`

  Use the actor's online network to get the action mean values using the current states as the inputs.
  Then, use the critic online network in order to get the gradients of the critic output with respect to the
  action mean values :math:`\nabla _a Q(s,a)|_{s=s_t,a=\mu(s_t ) }`.
  Using the chain rule, calculate the gradients of the actor's output, with respect to the actor weights,
  given :math:`\nabla_a Q(s,a)`. Finally, apply those gradients to the actor network.

After every training step, do a soft update of the critic and actor target networks' weights from the online networks.


.. autoclass:: rl_coach.agents.ddpg_agent.DDPGAlgorithmParameters