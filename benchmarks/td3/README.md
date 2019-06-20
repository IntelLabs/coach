# Twin Delayed DDPG

Each experiment uses 5 seeds and is trained for 1M environment steps.
The parameters used for TD3 are the same parameters as described in the [original paper](https://arxiv.org/pdf/1802.09477.pdf), and [repository](https://github.com/sfujim/TD3). 

### Ant TD3 - single worker

```bash
coach -p Mujoco_TD3 -lvl ant
```

<img src="ant.png" alt="Ant TD3" width="800"/>


### Hopper TD3 - single worker

```bash
coach -p Mujoco_TD3 -lvl hopper
```

<img src="hopper.png" alt="Hopper TD3" width="800"/>


### Half Cheetah TD3 - single worker

```bash
coach -p Mujoco_TD3 -lvl half_cheetah
```

<img src="half_cheetah.png" alt="Half Cheetah TD3" width="800"/>


### Reacher TD3 - single worker

```bash
coach -p Mujoco_TD3 -lvl reacher
```

<img src="reacher.png" alt="Reacher TD3" width="800"/>


### Walker2D TD3 - single worker

```bash
coach -p Mujoco_TD3 -lvl walker2d
```

<img src="walker2d.png" alt="Walker2D TD3" width="800"/>
