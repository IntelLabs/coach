# Coach Benchmarks

The following table represents the current status of algorithms implemented in Coach relative to the results reported in the original papers. The detailed results for each algorithm can be seen by clicking on its name.

The X axis in all the figures is the total steps (for multi-threaded runs, this is the number of steps per worker).
The Y axis in all the figures is the average episode reward with an averaging window of 100 timesteps.

For each algorithm, there is a command line for reproducing the results of each graph.
These are the results you can expect to get when running the pre-defined presets in Coach.

The environments that were used for testing include:
* **Atari** - Breakout, Pong and Space Invaders
* **Mujoco** - Inverted Pendulum, Inverted Double Pendulum, Reacher, Hopper, Half Cheetah, Walker 2D, Ant, Swimmer and Humanoid.
* **Doom** - Basic, Health Gathering (D1: Basic), Health Gathering Supreme (D2: Navigation), Battle (D3: Battle)
* **Fetch** - Reach, Slide, Push, Pick-and-Place

## Summary

![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) *Reproducing paper's results*

![#ceffad](https://placehold.it/15/ceffad/000000?text=+) *Reproducing paper's results for some of the environments*

![#FFA500](https://placehold.it/15/FFA500/000000?text=+) *Training but not reproducing paper's results*

![#FF4040](https://placehold.it/15/FF4040/000000?text=+) *Not training*



|                         |**Status**                                                |**Environments**|**Comments**|
| ----------------------- |:--------------------------------------------------------:|:--------------:|:--------:|
|**[DQN](dqn)**                  | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           |  |
|**[Dueling DDQN](dueling_ddqn)**| ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           |  |
|**[Dueling DDQN with PER](dueling_ddqn_with_per)**| ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           | |
|**[Bootstrapped DQN](bootstrapped_dqn)**| ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           | |
|**[QR-DQN](qr_dqn)**            | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           | |
|**[A3C](a3c)**                  | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari, Mujoco   | |
|**[Clipped PPO](clipped_ppo)**  | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Mujoco          | |
|**[DDPG](ddpg)**                | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Mujoco          | |
|**[NEC](nec)**                  | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Atari           | |
|**[HER](ddpg_her)**                  | ![#2E8B57](https://placehold.it/15/2E8B57/000000?text=+) |Fetch           | |
|**[HAC](hac)**                  | ![#969696](https://placehold.it/15/969696/000000?text=+) |Pendulum        | |
|**[DFP](dfp)**                  | ![#ceffad](https://placehold.it/15/ceffad/000000?text=+) |Doom            | Doom Battle was not verified |


**Click on each algorithm to see detailed benchmarking results**
