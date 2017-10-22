# N-Step Q Learning

**Actions space:** Discrete

**References:** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Network Structure

<p style="text-align: center;">

<img src="..\..\design_imgs\dqn.png">

</p>



## Algorithm Description

### Training the network

The $N$-step Q learning algorithm works in similar manner to DQN except for the following changes:

1. No replay buffer is used. Instead of sampling random batches of transitions, the network is trained every $N$ steps using the latest $N$ steps played by the agent.

2. In order to stabilize the learning, multiple workers work together to update the network. This creates the same effect as uncorrelating the samples used for training.

3. Instead of using single-step Q targets for the network, the rewards from $N$ consequent steps are accumulated to form the $N$-step Q targets, according to the following equation: 
$$R(s_t, a_t) = \sum_{i=t}^{i=t + k - 1} \gamma^{i-t}r_i +\gamma^{k} V(s_{t+k})$$
where $k$ is $T_{max} - State\_Index$ for each state in the batch

