# DDPG

Each experiment uses 3 seeds and is trained for 2k environment steps.
The parameters used for DDPG are the same parameters as described in the [original paper](https://arxiv.org/abs/1509.02971).

### Inverted Pendulum DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl inverted_pendulum
```

<img src="inverted_pendulum_ddpg.png" alt="Inverted Pendulum DDPG" width="800"/>


### Inverted Double Pendulum DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl inverted_double_pendulum
```

<img src="inverted_double_pendulum_ddpg.png" alt="Inverted Double Pendulum DDPG" width="800"/>


### Reacher DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl reacher
```

<img src="reacher_ddpg.png" alt="Reacher DDPG" width="800"/>


### Hopper DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl hopper
```

<img src="hopper_ddpg.png" alt="Hopper DDPG" width="800"/>


### Half Cheetah DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl half_cheetah
```

<img src="half_cheetah_ddpg.png" alt="Half Cheetah DDPG" width="800"/>


### Walker 2D DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl walker2d
```

<img src="walker2d_ddpg.png" alt="Walker 2D DDPG" width="800"/>


### Ant DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl ant
```

<img src="ant_ddpg.png" alt="Ant DDPG" width="800"/>


### Swimmer DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl swimmer
```

<img src="swimmer_ddpg.png" alt="Swimmer DDPG" width="800"/>


### Humanoid DDPG - single worker

```bash
coach -p Mujoco_DDPG -lvl humanoid
```

<img src="humanoid_ddpg.png" alt="Humanoid DDPG" width="800"/>
