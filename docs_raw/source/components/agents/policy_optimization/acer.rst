ACER
============

**Actions space:** Discrete

**References:** `Sample Efficient Actor-Critic with Experience Replay <https://arxiv.org/abs/1611.01224>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/acer.png
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
Each iteration perform one on-policy update with a batch of the last :math:`T_{max}` transitions,
and :math:`n` (replay ratio) off-policy updates from batches of :math:`T_{max}` transitions sampled from the replay buffer.

Each update perform the following procedure:

1. **Calculate state values:**

   .. math:: V(s_t) = \mathbb{E}_{a \sim \pi} [Q(s_t,a)]

2. **Calculate Q retrace:**

    .. math::   Q^{ret}(s_t,a_t) = r_t +\gamma \bar{\rho}_{t+1}[Q^{ret}(s_{t+1},a_{t+1}) - Q(s_{t+1},a_{t+1})] + \gamma V(s_{t+1})
    .. math::   \text{where} \quad \bar{\rho}_{t} = \min{\left\{c,\rho_t\right\}},\quad \rho_t=\frac{\pi (a_t \mid s_t)}{\mu (a_t \mid s_t)}

3. **Accumulate gradients:**

    :math:`\bullet` **Policy gradients (with bias correction):**

        .. math::  \hat{g}_t^{policy} & = & \bar{\rho}_{t} \nabla \log \pi (a_t \mid s_t) [Q^{ret}(s_t,a_t) - V(s_t)] \\
                    & & + \mathbb{E}_{a \sim \pi} \left(\left[\frac{\rho_t(a)-c}{\rho_t(a)}\right] \nabla \log \pi (a \mid s_t) [Q(s_t,a) - V(s_t)] \right)

    :math:`\bullet` **Q-Head gradients (MSE):**

        .. math::  \hat{g}_t^{Q} = (Q^{ret}(s_t,a_t) - Q(s_t,a_t)) \nabla Q(s_t,a_t)\\

4. **(Optional) Trust region update:** change the policy loss gradient w.r.t network output:

    .. math::  \hat{g}_t^{trust-region} = \hat{g}_t^{policy} - \max \left\{0, \frac{k^T \hat{g}_t^{policy} - \delta}{\lVert k \rVert_2^2}\right\} k
    .. math::  \text{where} \quad k = \nabla D_{KL}[\pi_{avg} \parallel \pi]

    The average policy network is an exponential moving average of the parameters of the network (:math:`\theta_{avg}=\alpha\theta_{avg}+(1-\alpha)\theta`).
    The goal of the trust region update is to the difference between the updated policy and the average policy to ensure stability.



.. autoclass:: rl_coach.agents.acer_agent.ACERAlgorithmParameters