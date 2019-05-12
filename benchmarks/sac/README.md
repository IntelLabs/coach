# Soft Actor Critic

Each experiment uses 3 seeds and is trained for 3M environment steps.
The parameters used for SAC are the same parameters as described in the [original paper](https://arxiv.org/abs/1801.01290).

### Inverted Pendulum SAC - single worker

```bash
coach -p Mujoco_SAC -lvl inverted_pendulum
```

<img src="inverted_pendulum_sac.png" alt="Inverted Pendulum SAC" width="800"/>


### Hopper Clipped SAC - single worker

```bash
coach -p Mujoco_SAC -lvl hopper
```

<img src="hopper_sac.png" alt="Hopper SAC" width="800"/>


### Half Cheetah Clipped SAC - single worker

```bash
coach -p Mujoco_SAC -lvl half_cheetah
```

<img src="half_cheetah_sac.png" alt="Half Cheetah SAC" width="800"/>


### Walker 2D Clipped SAC - single worker

```bash
coach -p Mujoco_SAC -lvl walker2d
```

<img src="walker2d_sac.png" alt="Walker 2D SAC" width="800"/>


### Humanoid Clipped SAC - single worker

```bash
coach -p Mujoco_SAC -lvl humanoid
```

<img src="humanoid_sac.png" alt="Humanoid SAC" width="800"/>
