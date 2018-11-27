Neural Episodic Control
=======================

**Actions space:** Discrete

**References:** `Neural Episodic Control <https://arxiv.org/abs/1703.01988>`_

Network Structure
-----------------

.. image:: /_static/img/design_imgs/nec.png
   :width: 500px
   :align: center

Algorithm Description
---------------------
Choosing an action
++++++++++++++++++

1. Use the current state as an input to the online network and extract the state embedding, which is the intermediate
   output from the middleware.

2. For each possible action :math:`a_i`, run the DND head using the state embedding and the selected action :math:`a_i` as inputs.
   The DND is queried and returns the :math:`P` nearest neighbor keys and values. The keys and values are used to calculate
   and return the action :math:`Q` value from the network.

3. Pass all the :math:`Q` values to the exploration policy and choose an action accordingly.

4. Store the state embeddings and actions taken during the current episode in a small buffer :math:`B`, in order to
   accumulate transitions until it is possible to calculate the total discounted returns over the entire episode.

Finalizing an episode
+++++++++++++++++++++
For each step in the episode, the state embeddings and the taken actions are stored in the buffer :math:`B`.
When the episode is finished, the replay buffer calculates the :math:`N`-step total return of each transition in the
buffer, bootstrapped using the maximum :math:`Q` value of the :math:`N`-th transition. Those values are inserted
along with the total return into the DND, and the buffer :math:`B` is reset.

Training the network
++++++++++++++++++++
Train the network only when the DND has enough entries for querying.

To train the network, the current states are used as the inputs and the :math:`N`-step returns are used as the targets.
The :math:`N`-step return used takes into account :math:`N` consecutive steps, and bootstraps the last value from
the network if necessary:
:math:`y_t=\sum_{j=0}^{N-1}\gamma^j r(s_{t+j},a_{t+j} ) +\gamma^N   max_a Q(s_{t+N},a)`



.. autoclass:: rl_coach.agents.nec_agent.NECAlgorithmParameters
