# Policy Gradient

**Actions space:** Discrete|Continuous

**References:** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

## Network Structure

<p style="text-align: center;">

<img src="..\..\design_imgs\pg.png">

</p>

## Algorithm Description
### Choosing an action - Discrete actions
Run the current states through the network and get a policy distribution over the actions. While training, sample from the policy distribution. When testing, take the action with the highest probability. 

### Training the network
The policy head loss is defined as $ L=-log (\pi) \cdot  PolicyGradientRescaler $. The $PolicyGradientRescaler$ is used in order to reduce the policy gradient variance, which might be very noisy. This is done in order to reduce the variance of the updates, since noisy gradient updates might destabilize the policy's convergence. The rescaler is a configurable parameter and there are few options to choose from:	
* **Total Episode Return** - The sum of all the discounted rewards during the episode.
* **Future Return** - Return from each transition until the end of the episode.
* **Future Return Normalized by Episode** - Future returns across the episode normalized by the episode's mean and standard deviation.
* **Future Return Normalized by Timestep** - Future returns normalized using running means and standard deviations, which are calculated seperately for each timestep, across different episodes. 

Gradients are accumulated over a number of full played episodes. The gradients accumulation over several episodes serves the same purpose - reducing the update variance. After accumulating gradients for several episodes, the gradients are then applied to the network. 

