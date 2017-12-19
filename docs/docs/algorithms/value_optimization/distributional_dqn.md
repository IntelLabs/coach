# Distributional DQN

**Actions space:** Discrete

**References:** [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## Network Structure

<p style="text-align: center;">

<img src="..\..\design_imgs\distributional_dqn.png">

</p>



## Algorithmic Description

### Training the network

1. Sample a batch of transitions from the replay buffer. 
2. The Bellman update is projected to the set of atoms representing the $ Q $ values distribution, such that the $i-th$ component of the projected update is calculated as follows:
   $$ (\Phi \hat{T} Z_{\theta}(s_t,a_t))_i=\sum_{j=0}^{N-1}\Big[1-\frac{|[\hat{T}_{z_{j}}]^{V_{MAX}}_{V_{MIN}}-z_i|}{\Delta z}\Big]^1_0 \ p_j(s_{t+1}, \pi(s_{t+1})) $$
   where:
   	*  $[ \cdot ] $ bounds its argument in the range [a, b]
   	*  $\hat{T}_{z_{j}}$ is the Bellman update for atom $z_j$: &nbsp; &nbsp;   $\hat{T}_{z_{j}} := r+\gamma z_j$


3. Network is trained with the cross entropy loss between the resulting probability distribution and the target probability distribution.   Only the target of the actions that were actually taken is updated. 
4. Once in every few thousand steps, weights are copied from the online network to the target network.



