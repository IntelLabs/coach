# Quantile Regression DQN

Each experiment uses 3 seeds and is trained for 10k environment steps.
The parameters used for QR-DQN are the same parameters as described in the [original paper](https://arxiv.org/abs/1710.10044.pdf).

### Breakout QR-DQN - single worker

```bash
coach -p Atari_QR_DQN -lvl breakout
```

<img src="breakout_qr_dqn.png" alt="Breakout QR-DQN" width="800"/>


### Pong QR-DQN - single worker

```bash
coach -p Atari_QR_DQN -lvl pong
```

<img src="pong_qr_dqn.png" alt="Pong QR-DQN" width="800"/>
