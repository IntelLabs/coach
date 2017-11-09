# Coach Benchmarks

The following figures are training curves of some of the presets available through Coach.
The X axis in all the figures is the total steps (for multi-threaded runs, this is the accumulated number of steps over all the workers).
The Y axis in all the figures is the average episode reward with an averaging window of 11 episodes.
These are the result you can expect to get when running the pre-defined presets in Coach.


## A3C

### Breakout_A3C with 16 workers

```bash
python3 coach.py -p Breakout_A3C -n 16 -r
```

<img src="img/Breakout_A3C_16_workers.png" alt="Breakout_A3C_16_workers" width="800"/>

### InvertedPendulum_A3C with 16 workers

```bash
python3 coach.py -p InvertedPendulum_A3C -n 16 -r
```

<img src="img/Inverted_Pendulum_A3C_16_workers.png" alt="Inverted_Pendulum_A3C_16_workers" width="800"/>

### Hopper_A3C with 16 workers

```bash
python3 coach.py -p Hopper_A3C -n 16 -r
```

<img src="img/Hopper_A3C_16_workers.png" alt="Hopper_A3C_16_workers" width="800"/>

### Ant_A3C with 16 workers

```bash
python3 coach.py -p Ant_A3C -n 16 -r
```

<img src="img/Ant_A3C_16_workers.png" alt="Ant_A3C_16_workers" width="800"/>

## Clipped PPO

### InvertedPendulum_ClippedPPO with 16 workers

```bash
python3 coach.py -p InvertedPendulum_ClippedPPO -n 16 -r
```

<img src="img/InvertedPendulum_ClippedPPO_16_workers.png" alt="InvertedPendulum_ClippedPPO_16_workers" width="800"/>

### Hopper_ClippedPPO with 16 workers

```bash
python3 coach.py -p Hopper_ClippedPPO -n 16 -r
```

<img src="img/Hopper_ClippedPPO_16_workers.png" alt="Hopper_Clipped_PPO_16_workers" width="800"/>

### Humanoid_ClippedPPO with 16 workers

```bash
python3 coach.py -p Humanoid_ClippedPPO -n 16 -r
```

<img src="img/Humanoid_ClippedPPO_16_workers.png" alt="Humanoid_ClippedPPO_16_workers" width="800"/>

## DQN

### Pong_DQN

```bash
python3 coach.py -p Pong_DQN -r
```

<img src="img/Pong_DQN.png" alt="Pong_DQN" width="800"/>

### Doom_Basic_DQN

```bash
python3 coach.py -p Doom_Basic_DQN -r
```

<img src="img/Doom_Basic_DQN.png" alt="Doom_Basic_DQN" width="800"/>

## Dueling DDQN

### Doom_Basic_Dueling_DDQN

```bash
python3 coach.py -p Doom_Basic_Dueling_DDQN -r
```

<img src="img/Doom_Basic_Dueling_DDQN.png" alt="Doom_Basic_Dueling_DDQN" width="800"/>

## DFP

### Doom_Health_DFP

```bash
python3 coach.py -p Doom_Health_DFP -r
```

<img src="img/Doom_Health.png" alt="Doom_Health_DFP" width="800"/>

## MMC

### Doom_Health_MMC

```bash
python3 coach.py -p Doom_Health_MMC -r
```

<img src="img/Doom_Health_MMC.png" alt="Doom_Health_MMC" width="800"/>

## NEC

## Doom_Basic_NEC

```bash
python3 coach.py -p Doom_Basic_NEC -r
```

<img src="img/Doom_Basic_NEC.png" alt="Doom_Basic_NEC" width="800"/>

## PG

### CartPole_PG

```bash
python3 coach.py -p CartPole_PG -r
```

<img src="img/CartPole_PG.png" alt="CartPole_PG" width="800"/>

## DDPG

### Pendulum_DDPG

```bash
python3 coach.py -p Pendulum_DDPG -r
```

<img src="img/Pendulum_DDPG.png" alt="Pendulum_DDPG" width="800"/>


## NAF

### InvertedPendulum_NAF

```bash
python3 coach.py -p InvertedPendulum_NAF -r
```

<img src="img/InvertedPendulum_NAF.png" alt="InvertedPendulum_NAF" width="800"/>

### Pendulum_NAF

```bash
python3 coach.py -p Pendulum_NAF -r
```

<img src="img/Pendulum_NAF.png" alt="Pendulum_NAF" width="800"/>
