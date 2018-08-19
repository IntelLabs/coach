# DQN

Each experiment uses 3 seeds.
The parameters used for DQN are the same parameters as described in the [original paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), except for the optimizer (changed to ADAM) and learning rate (1e-4) used.

### Breakout DQN - single worker

```bash
coach -p Atari_DQN -lvl breakout
```

<img src="breakout_dqn.png" alt="Breakout DQN" width="800"/>

### Pong DQN - single worker

```bash
coach -p Atari_DQN -lvl pong
```

<img src="pong_dqn.png" alt="Pong DQN" width="800"/>

### Space Invaders DQN - single worker

```bash
coach -p Atari_DQN -lvl space_invaders
```

<img src="space_invaders_dqn.png" alt="Space Invaders DQN" width="800"/>


