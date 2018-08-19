# Bootstrapped DQN

Each experiment uses 3 seeds.
The parameters used for Bootstrapped DQN are the same parameters as described in the [original paper](https://arxiv.org/abs/1602.04621.pdf).

### Breakout Bootstrapped DQN - single worker

```bash
coach -p Atari_Bootstrapped_DQN -lvl breakout
```

<img src="breakout_bootstrapped_dqn.png" alt="Breakout Bootstrapped DQN" width="800"/>


### Pong Bootstrapped DQN - single worker

```bash
coach -p Atari_Bootstrapped_DQN -lvl pong
```

<img src="pong_bootstrapped_dqn.png" alt="Pong Bootstrapped DQN" width="800"/>


### Space Invaders Bootstrapped DQN - single worker

```bash
coach -p Atari_Bootstrapped_DQN -lvl space_invaders
```

<img src="space_invaders_bootstrapped_dqn.png" alt="Space Invaders Bootstrapped DQN" width="800"/>

