Twin Delayed Deep Deterministic Policy Gradient
==================================

**Actions space:** Continuous

**References:** `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/pdf/1802.09477>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/td3.png
   :align: center

Algorithm Description
---------------------
Choosing an action
++++++++++++++++++

Pass the current states through the actor network, and get an action mean vector :math:`\mu`.
While in training phase, use a continuous exploration policy, such as a small zero-meaned gaussian noise,
to add exploration noise to the action. When testing, use the mean vector :math:`\mu` as-is.

Training the network
++++++++++++++++++++

Start by sampling a batch of transitions from the experience replay.

* To train the two **critic networks**, use the following targets:

  :math:`y_t=r(s_t,a_t )+\gamma \cdot \min_{i=1,2} Q_{i}(s_{t+1},\mu(s_{t+1} )+[\mathcal{N}(0,\,\sigma^{2})]^{MAX\_NOISE}_{MIN\_NOISE})`

  First run the actor target network, using the next states as the inputs, and get :math:`\mu (s_{t+1} )`. Then, add a
  clipped gaussian noise to these actions, and clip the resulting actions to the actions space.
  Next, run the critic target networks using the next states and :math:`\mu (s_{t+1} )+[\mathcal{N}(0,\,\sigma^{2})]^{MAX\_NOISE}_{MIN\_NOISE}`,
  and use the minimum between the two critic networks predictions in order to calculate :math:`y_t` according to the
  equation above. To train the networks, use the current states and actions as the inputs, and :math:`y_t`
  as the targets.

* To train the **actor network**, use the following equation:

  :math:`\nabla_{\theta^\mu } J \approx E_{s_t \tilde{} \rho^\beta } [\nabla_a Q_{1}(s,a)|_{s=s_t,a=\mu (s_t ) } \cdot \nabla_{\theta^\mu} \mu(s)|_{s=s_t} ]`

  Use the actor's online network to get the action mean values using the current states as the inputs.
  Then, use the first critic's online network in order to get the gradients of the critic output with respect to the
  action mean values :math:`\nabla _a Q_{1}(s,a)|_{s=s_t,a=\mu(s_t ) }`.
  Using the chain rule, calculate the gradients of the actor's output, with respect to the actor weights,
  given :math:`\nabla_a Q(s,a)`. Finally, apply those gradients to the actor network.

  The actor's training is done at a slower frequency than the critic's training, in order to allow the critic to better fit the
  current policy, before exercising the critic in order to train the actor.
  Following the same, delayed, actor's training cadence, do a soft update of the critic and actor target networks' weights
  from the online networks.


.. autoclass:: rl_coach.agents.td3_agent.TD3AlgorithmParameters