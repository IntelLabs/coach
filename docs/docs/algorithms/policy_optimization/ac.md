# Actor-Critic

**Actions space:** Discrete|Continuous

**References:** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Network Structure 
<p style="text-align: center;">
<img src="..\..\design_imgs\ac.png" width=500>
</p>
## Algorithm Description

### Choosing an action - Discrete actions

The policy network is used in order to predict action probabilites. While training, a sample is taken from a categorical distribution assigned with these probabilities. When testing, the action with the highest probability is used.

### Training the network
A batch of $ T_{max} $ transitions is used, and the advantages are calculated upon it.

Advantages can be calculated by either of the following methods (configured by the selected preset) -

1. **A_VALUE** - Estimating advantage directly:$$ A(s_t, a_t) = \underbrace{\sum_{i=t}^{i=t + k - 1} \gamma^{i-t}r_i +\gamma^{k} V(s_{t+k})}_{Q(s_t, a_t)} - V(s_t) $$where $k$ is $T_{max} - State\_Index$ for each state in the batch.
2. **GAE** - By following the [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) paper. 

The advantages are then used in order to accumulate gradients according to 
$$ L = -\mathop{\mathbb{E}} [log (\pi) \cdot A] $$

