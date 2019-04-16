# ACER

Each experiment uses 3 seeds.
The parameters used for ACER are the same parameters as described in the [original paper](https://arxiv.org/abs/1611.01224), except for the optimizer (changed to ADAM) and learning rate (1e-4) used.

### Breakout ACER - 16 workers

```bash
coach -p Atari_ACER -lvl breakout -n 16
```

<img src="breakout_acer_16_workers.png" alt="Breakout ACER" width="800"/>

### Space Invaders ACER - 16 workers

```bash
coach -p Atari_ACER -lvl space_invaders -n 16
```

<img src="space_invaders_acer_16_workers.png" alt="Space Invaders ACER" width="800"/>

### Pong ACER - 16 workers

```bash
coach -p Atari_ACER -lvl pong -n 16
```

<img src="pong_acer_16_workers.png" alt="Pong ACER" width="800"/>
