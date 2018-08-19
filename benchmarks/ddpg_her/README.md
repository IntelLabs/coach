# DDPG with Hindsight Experience Replay

Each experiment uses 3 seeds.
The parameters used for DDPG HER are the same parameters as described in the [following paper](https://arxiv.org/abs/1802.09464).

### Fetch Reach DDPG HER - single worker

```bash
coach -p Fetch_DDPG_HER_baselines -lvl reach
```

<img src="fetch_ddpg_her_reach_1_worker.png" alt="Fetch DDPG HER Reach 1 Worker" width="800"/>


### Fetch Push DDPG HER - 8 workers

```bash
coach -p Fetch_DDPG_HER_baselines -lvl push -n 8
```

<img src="fetch_ddpg_her_push_8_workers.png" alt="Fetch DDPG HER Push 8 Worker" width="800"/>


### Fetch Slide DDPG HER - 8 workers

```bash
coach -p Fetch_DDPG_HER_baselines -lvl slide -n 8
```

<img src="fetch_ddpg_her_slide_8_workers.png" alt="Fetch DDPG HER Slide 8 Worker" width="800"/>


### Fetch Pick And Place DDPG HER - 8 workers

```bash
coach -p Fetch_DDPG_HER -lvl pick_and_place -n 8
```

<img src="fetch_ddpg_her_pick_and_place_8_workers.png" alt="Fetch DDPG HER Pick And Place 8 Workers" width="800"/>

