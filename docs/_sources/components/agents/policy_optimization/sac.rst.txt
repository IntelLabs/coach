Soft Actor-Critic
============

**Actions space:** Continuous

**References:** `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/sac.png
   :align: center

Algorithm Description
---------------------

Choosing an action - Continuous actions
+++++++++++++++++++++++++++++++++++++

The policy network is used in order to predict mean and log std for each action. While training, a sample is taken
from a Gaussian distribution with these mean and std values. When testing, the agent can choose deterministically
by picking the mean value or sample from a gaussian distribution like in training.

Training the network
++++++++++++++++++++
Start by sampling a batch :math:`B` of transitions from the experience replay.

* To train the **Q network**, use the following targets:

  .. math:: y_t^Q=r(s_t,a_t)+\gamma \cdot V(s_{t+1})

  The state value used in the above target is acquired by running the target state value network.

* To train the **State Value network**, use the following targets:

  .. math:: y_t^V = \min_{i=1,2}Q_i(s_t,\tilde{a}) - log\pi (\tilde{a} \vert s),\,\,\,\, \tilde{a} \sim \pi(\cdot \vert s_t)

  The state value network is trained using a sample-based approximation of the connection between and state value and state
  action values, The actions used for constructing the target are **not** sampled from the replay buffer, but rather sampled
  from the current policy.

* To train the **actor network**, use the following equation:

  .. math:: \nabla_{\theta} J \approx \nabla_{\theta} \frac{1}{\vert B \vert} \sum_{s_t\in B} \left( Q \left(s_t, \tilde{a}_\theta(s_t)\right) - log\pi_{\theta}(\tilde{a}_{\theta}(s_t)\vert s_t) \right),\,\,\,\, \tilde{a} \sim \pi(\cdot \vert s_t)

After every training step, do a soft update of the V target network's weights from the online networks.


.. autoclass:: rl_coach.agents.soft_actor_critic_agent.SoftActorCriticAlgorithmParameters