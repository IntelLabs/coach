# Neural Episodic Control

**Actions space:** Discrete

**References:** [Neural Episodic Control](https://arxiv.org/abs/1703.01988)

## Network Structure

<p style="text-align: center;">

<img src="..\..\design_imgs\nec.png" width=500>

</p>

## Algorithm Description
### Choosing an action
1. Use the current state as an input to the online network and extract the state embedding, which is the intermediate output from the middleware. 
2. For each possible action $a_i$, run the DND head using the state embedding and the selected action $a_i$ as inputs. The DND is queried and returns the $ P $ nearest neighbor keys and values. The keys and values are used to calculate and return the action $ Q $ value from the network. 
3. Pass all the $ Q $ values to the exploration policy and choose an action accordingly. 
4. Store the state embeddings and actions taken during the current episode in a small buffer $B$, in order to accumulate transitions until it is possible to calculate the total discounted returns over the entire episode.

### Finalizing an episode
For each step in the episode, the state embeddings and the taken actions are stored in the buffer $B$. When the episode is finished, the replay buffer calculates the $ N $-step total return of each transition in the buffer, bootstrapped using the maximum $Q$ value of the $N$-th transition. Those values are inserted along with the total return into the DND, and the buffer $B$ is reset.
### Training the network
Train the network only when the DND has enough entries for querying.

To train the network, the current states are used as the inputs and the $N$-step returns are used as the targets. The $N$-step return used takes into account $ N $ consecutive steps, and bootstraps the last value from the network if necessary:
$$ y_t=\sum_{j=0}^{N-1}\gamma^j r(s_{t+j},a_{t+j} ) +\gamma^N   max_a Q(s_{t+N},a) $$
