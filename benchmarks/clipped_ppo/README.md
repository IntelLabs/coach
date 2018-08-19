# Clipped PPO

Each experiment uses 3 seeds and is trained for 10k environment steps.
The parameters used for Clipped PPO are the same parameters as described in the [original paper](https://arxiv.org/abs/1707.06347).

### Inverted Pendulum Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl inverted_pendulum
```

<img src="inverted_pendulum_clipped_ppo.png" alt="Inverted Pendulum Clipped PPO" width="800"/>


### Inverted Double Pendulum Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl inverted_double_pendulum
```

<img src="inverted_double_pendulum_clipped_ppo.png" alt="Inverted Double Pendulum Clipped PPO" width="800"/>


### Reacher Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl reacher
```

<img src="reacher_clipped_ppo.png" alt="Reacher Clipped PPO" width="800"/>


### Hopper Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl hopper
```

<img src="hopper_clipped_ppo.png" alt="Hopper Clipped PPO" width="800"/>


### Half Cheetah Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl half_cheetah
```

<img src="half_cheetah_clipped_ppo.png" alt="Half Cheetah Clipped PPO" width="800"/>


### Walker 2D Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl walker2d
```

<img src="walker2d_clipped_ppo.png" alt="Walker 2D Clipped PPO" width="800"/>


### Ant Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl ant
```

<img src="ant_clipped_ppo.png" alt="Ant Clipped PPO" width="800"/>


### Swimmer Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl swimmer
```

<img src="swimmer_clipped_ppo.png" alt="Swimmer Clipped PPO" width="800"/>


### Humanoid Clipped PPO - single worker

```bash
coach -p Mujoco_ClippedPPO -lvl humanoid
```

<img src="humanoid_clipped_ppo.png" alt="Humanoid Clipped PPO" width="800"/>
