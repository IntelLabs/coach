# Double DQN

**Actions space:** Discrete

**References:** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461.pdf)

## Network Structure

<p style="text-align: center;">

<img src="..\..\design_imgs\dqn.png">

</p>



## Algorithm Description

### Training the network
1. Sample a batch of transitions from the replay buffer. 
2. Using the next states from the sampled batch, run the online network in order to find the $Q$ maximizing action $argmax_a Q(s_{t+1},a)$. For these actions, use the corresponding next states and run the target network to calculate $Q(s_{t+1},argmax_a Q(s_{t+1},a))$.
3. In order to zero out the updates for the actions that were not played (resulting from zeroing the MSE loss), use the current states from the sampled batch, and run the online network to get the current Q values predictions. Set those values as the targets for the actions that were not actually played. 
4. For each action that was played, use the following equation for calculating the targets of the network:
   $$ y_t=r(s_t,a_t )+\gamma \cdot Q(s_{t+1},argmax_a Q(s_{t+1},a)) $$


5. Finally, train the online network using the current states as inputs, and with the aforementioned targets. 
6. Once in every few thousand steps, copy the weights from the online network to the target network.