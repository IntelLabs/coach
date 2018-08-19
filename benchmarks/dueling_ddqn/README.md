# Dueling DDQN

Each experiment uses 3 seeds and is trained for 10k environment steps.
The parameters used for Dueling DDQN are the same parameters as described in the [original paper](https://arxiv.org/abs/1706.01502).

### Pong Dueling DDQN - single worker

```bash
coach -p Atari_Dueling_DDQN -lvl pong
```

<img src="pong_dueling_ddqn.png" alt="Pong Dueling DDQN" width="800"/>


### Breakout Dueling DDQN - single worker

```bash
coach -p Atari_Dueling_DDQN -lvl breakout
```

<img src="breakout_dueling_ddqn.png" alt="Breakout Dueling DDQN" width="800"/>


### Space Invaders Dueling DDQN - single worker

```bash
coach -p Atari_Dueling_DDQN -lvl space_invaders
```

<img src="space_invaders_dueling_ddqn.png" alt="Space Invaders Dueling DDQN" width="800"/>





