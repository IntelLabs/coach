# Clipped Proximal Policy Optimization

**Actions space:** Discrete|Continuous

**References:** [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

## Network Structure

<p style="text-align: center;">
<img src="..\..\design_imgs\ppo.png">
</p>


## Algorithm Description
### Choosing an action - Continuous action
Same as in PPO. 
### Training the network
Very similar to PPO, with several small (but very simplifying) changes:

1. Train both the value and policy networks, simultaneously, by defining a single loss function, which is the sum of each of the networks loss functions. Then, back propagate gradients only once from this unified loss function.

2. The unified network's optimizer is set to Adam (instead of L-BFGS for the value network as in PPO). 

3. Value targets are now also calculated based on the GAE advantages. In this method, the $ V $ values are predicted from the critic network, and then added to the GAE based advantages, in order to get a $ Q $ value for each action. Now, since our critic network is predicting a $ V $ value for each state, setting the $ Q $ calculated action-values as a target, will on average serve as a $ V $ state-value target.  

4. Instead of adapting the penalizing KL divergence coefficient used in PPO, the likelihood ratio $r_t(\theta) =\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}$ is clipped, to achieve a similar effect. This is done by defining the policy's loss function to be the minimum between the standard surrogate loss and an epsilon clipped surrogate loss:

$$L^{CLIP}(\theta)=E_{t}[min(r_t(\theta)\cdot \hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t)]  $$