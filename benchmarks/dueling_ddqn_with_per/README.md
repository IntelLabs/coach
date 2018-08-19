# Dueling DDQN with Prioritized Experience Replay

Each experiment uses 3 seeds and is trained for 10k environment steps.
The parameters used for Dueling DDQN with PER are the same parameters as described in the [following paper](https://arxiv.org/abs/1511.05952).

### Breakout Dueling DDQN with PER - single worker

```bash
coach -p Atari_Dueling_DDQN_with_PER_OpenAI -lvl breakout
```

<img src="breakout_dueling_ddqn_with_per.png" alt="Breakout Dueling DDQN with PER" width="800"/>


### Pong Dueling DDQN with PER - single worker

```bash
coach -p Atari_Dueling_DDQN_with_PER_OpenAI -lvl pong
```

<img src="pong_dueling_ddqn_with_per.png" alt="Pong Dueling DDQN with PER" width="800"/>


### Space Invaders Dueling DDQN with PER - single worker

```bash
coach -p Atari_Dueling_DDQN_with_PER_OpenAI -lvl space_invaders
```

<img src="space_invaders_dueling_ddqn_with_per.png" alt="Space Invaders Dueling DDQN with PER" width="800"/>

