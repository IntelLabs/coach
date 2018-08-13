# A3C

Each experiment uses 3 seeds.
The parameters used for Clipped PPO are the same parameters as described in the [original paper](https://arxiv.org/abs/1707.06347).

### Inverted Pendulum A3C - 1/2/4/8/16 workers

```bash
python3 coach.py -p Mujoco_A3C -lvl inverted_pendulum -n 1
python3 coach.py -p Mujoco_A3C -lvl inverted_pendulum -n 2
python3 coach.py -p Mujoco_A3C -lvl inverted_pendulum -n 4
python3 coach.py -p Mujoco_A3C -lvl inverted_pendulum -n 8
python3 coach.py -p Mujoco_A3C -lvl inverted_pendulum -n 16
```

<img src="inverted_pendulum_a3c.png" alt="Inverted Pendulum A3C" width="800"/>


### Hopper A3C - 16 workers

```bash
python3 coach.py -p Mujoco_A3C -lvl hopper -n 16
```

<img src="hopper_a3c_16_workers.png" alt="Hopper A3C 16 workers" width="800"/>


### Walker2D A3C - 16 workers

```bash
python3 coach.py -p Mujoco_A3C -lvl walker2d -n 16
```

<img src="walker2d_a3c_16_workers.png" alt="Walker2D A3C 16 workers" width="800"/>


### Space Invaders A3C - 16 workers

```bash
python3 coach.py -p Atari_A3C -lvl space_invaders -n 16
```

<img src="space_invaders_a3c_16_workers.png" alt="Space Invaders A3C 16 workers" width="800"/>
