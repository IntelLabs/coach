# DQN

Each experiment uses 3 seeds.
The parameters used for DQN are the same parameters as described in the [original paper](https://arxiv.org/abs/1607.05077.pdf).

### Breakout DQN - single worker

```bash
python3 coach.py -p Atari_DQN -lvl breakout
```

<img src="breakout_dqn.png" alt="Breakout DQN" width="800"/>

### Pong DQN - single worker

```bash
python3 coach.py -p Atari_DQN -lvl pong
```

<img src="pong_dqn.png" alt="Pong DQN" width="800"/>

### Space Invaders DQN - single worker

```bash
python3 coach.py -p Atari_DQN -lvl space_invaders
```

<img src="space_invaders_dqn.png" alt="Space Invaders DQN" width="800"/>


