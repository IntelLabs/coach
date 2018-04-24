# Persistent Advantage Learning

**Actions space:** Discrete

**References:** [Increasing the Action Gap: New Operators for Reinforcement Learning](https://arxiv.org/abs/1512.04860)

## Network Structure

<p style="text-align: center;">

<img src="../../design_imgs/dqn.png">

</p> 

## Algorithm Description
### Training the network
1. Sample a batch of transitions from the replay buffer. 

2. Start by calculating the initial target values in the same manner as they are calculated in DDQN
   $$ y_t^{DDQN}=r(s_t,a_t )+\gamma Q(s_{t+1},argmax_a Q(s_{t+1},a)) $$
3. The action gap $ V(s_t )-Q(s_t,a_t) $ should then be subtracted from each of the calculated targets. To calculate the action gap, run the target network using the current states and get the $ Q $ values for all the actions. Then estimate $ V $ as the maximum predicted $ Q $ value for the current state:
   $$ V(s_t )=max_a Q(s_t,a) $$
4. For _advantage learning (AL)_, reduce the action gap weighted by a predefined parameter $ \alpha $ from the targets $ y_t^{DDQN} $: 
   $$ y_t=y_t^{DDQN}-\alpha \cdot (V(s_t )-Q(s_t,a_t )) $$
5. For _persistent advantage learning (PAL)_, the target network is also used in order to calculate the action gap for the next state:
   $$ V(s_{t+1} )-Q(s_{t+1},a_{t+1}) $$
   where $ a_{t+1} $ is chosen by running the next states through the online network and choosing the action that has the highest predicted $ Q $ value. Finally, the targets will be defined as -
   $$ y_t=y_t^{DDQN}-\alpha \cdot min(V(s_t )-Q(s_t,a_t ),V(s_{t+1} )-Q(s_{t+1},a_{t+1} )) $$
6. Train the online network using the current states as inputs, and with the aforementioned targets.

7. Once in every few thousand steps, copy the weights from the online network to the target network.

