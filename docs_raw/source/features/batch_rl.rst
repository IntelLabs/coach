Batch Reinforcement Learning
============================

Coach supports Batch Reinforcement Learning, where learning is based solely on a (fixed) batch of data.
In Batch RL, we are given a dataset of experience, which was collected using some (one or more) deployed policies, and we would
like to use it to learn a better policy than what was used to collect the dataset.
There is no simulator to interact with, and so we cannot collect any new data, meaning we often cannot explore the MDP any further.
To make things even harder, we would also like to use the dataset in order to evaluate the newly learned policy
(using off-policy evaluation), since we do not have a simulator which we can use to evaluate the policy on.
Batch RL is also often beneficial in cases where we just want to separate the inference (data collection) from the
training process of a new policy. This is often the case where we have a system on which we could quite easily deploy a policy
and collect experience data, but cannot easily use that system's setup to online train a new policy (as is often the
case with more standard RL algorithms).

Coach supports (almost) all of the integrated off-policy algorithms with Batch RL.

A lot more details and example usage can be found in the
`tutorial <https://github.com/NervanaSystems/coach/blob/master/tutorials/4.%20Batch%20Reinforcement%20Learning.ipynb>`_.