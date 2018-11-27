Proximal Policy Optimization
============================

**Actions space:** Discrete | Continuous

**References:** `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/ppo.png
   :align: center


Algorithm Description
---------------------
Choosing an action - Continuous actions
+++++++++++++++++++++++++++++++++++++++
Run the observation through the policy network, and get the mean and standard deviation vectors for this observation.
While in training phase, sample from a multi-dimensional Gaussian distribution with these mean and standard deviation values.
When testing, just take the mean values predicted by the network.

Training the network
++++++++++++++++++++

1. Collect a big chunk of experience (in the order of thousands of transitions, sampled from multiple episodes).

2. Calculate the advantages for each transition, using the *Generalized Advantage Estimation* method (Schulman '2015).

3. Run a single training iteration of the value network using an L-BFGS optimizer. Unlike first order optimizers,
   the L-BFGS optimizer runs on the entire dataset at once, without batching.
   It continues running until some low loss threshold is reached. To prevent overfitting to the current dataset,
   the value targets are updated in a soft manner, using an Exponentially Weighted Moving Average, based on the total
   discounted returns of each state in each episode.

4. Run several training iterations of the policy network. This is done by using the previously calculated advantages as
   targets. The loss function penalizes policies that deviate too far from the old policy (the policy that was used *before*
   starting to run the current set of training iterations) using a regularization term.

5. After training is done, the last sampled KL divergence value will be compared with the *target KL divergence* value,
   in order to adapt the penalty coefficient used in the policy loss. If the KL divergence went too high,
   increase the penalty, if it went too low, reduce it. Otherwise, leave it unchanged.


.. autoclass:: rl_coach.agents.ppo_agent.PPOAlgorithmParameters