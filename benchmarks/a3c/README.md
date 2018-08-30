# A3C

Each experiment uses 3 seeds.
The parameters used for A3C are the same parameters as described in the [original paper](https://arxiv.org/abs/1602.01783).

### Inverted Pendulum A3C - 1/2/4/8/16 workers

```bash
coach -p Mujoco_A3C -lvl inverted_pendulum -n 1
coach -p Mujoco_A3C -lvl inverted_pendulum -n 2
coach -p Mujoco_A3C -lvl inverted_pendulum -n 4
coach -p Mujoco_A3C -lvl inverted_pendulum -n 8
coach -p Mujoco_A3C -lvl inverted_pendulum -n 16
```

<img src="inverted_pendulum_a3c.png" alt="Inverted Pendulum A3C" width="800"/>


### Hopper A3C - 16 workers

```bash
coach -p Mujoco_A3C -lvl hopper -n 16
```

<img src="hopper_a3c_16_workers.png" alt="Hopper A3C 16 workers" width="800"/>


### Walker2D A3C - 16 workers

```bash
coach -p Mujoco_A3C -lvl walker2d -n 16
```

<img src="walker2d_a3c_16_workers.png" alt="Walker2D A3C 16 workers" width="800"/>


### Half Cheetah A3C - 16 workers

```bash
coach -p Mujoco_A3C -lvl half_cheetah -n 16
```

<img src="half_cheetah_a3c_16_workers.png" alt="Half Cheetah A3C 16 workers" width="800"/>


### Ant A3C - 16 workers

```bash
coach -p Mujoco_A3C -lvl ant -n 16
```

<img src="ant_a3c_16_workers.png" alt="Ant A3C 16 workers" width="800"/>



### Space Invaders A3C - 16 workers

```bash
coach -p Atari_A3C -lvl space_invaders -n 16
```

<img src="space_invaders_a3c_16_workers.png" alt="Space Invaders A3C 16 workers" width="800"/>
