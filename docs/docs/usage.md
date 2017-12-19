# Coach Usage

## Training an Agent

### Single-threaded Algorithms

This is the most common case. Just choose a preset using the `-p` flag and press enter.

*Example:*

`python coach.py -p CartPole_DQN`

### Multi-threaded Algorithms

Multi-threaded algorithms are very common this days.
They typically achieve the best results, and scale gracefully with the number of threads.
In Coach, running such algorithms is done by selecting a suitable preset, and choosing the number of threads to run using the `-n` flag.

*Example:*

`python coach.py -p CartPole_A3C -n 8`

## Evaluating an Agent

There are several options for evaluating an agent during the training:

* For multi-threaded runs, an evaluation agent will constantly run in the background and evaluate the model during the training.

* For single-threaded runs, it is possible to define an evaluation period through the preset. This will run several episodes of evaluation once in a while.

Additionally, it is possible to save checkpoints of the agents networks and then run only in evaluation mode.
Saving checkpoints can be done by specifying the number of seconds between storing checkpoints using the `-s` flag.
The checkpoints will be saved into the experiment directory.
Loading a model for evaluation can be done by specifying the `-crd` flag with the experiment directory, and the `--evaluate` flag to disable training.

*Example:*

`python coach.py -p CartPole_DQN -s 60`
`python coach.py -p CartPole_DQN --evaluate -crd CHECKPOINT_RESTORE_DIR`

## Playing with the Environment as a Human

Interacting with the environment as a human can be useful for understanding its difficulties and for collecting data for imitation learning.
In Coach, this can be easily done by selecting a preset that defines the environment to use, and specifying the `--play` flag.
When the environment is loaded, the available keyboard buttons will be printed to the screen.
Pressing the escape key when finished will end the simulation and store the replay buffer in the experiment dir.

*Example:*

`python coach.py -p Breakout_DQN --play`

## Learning Through Imitation Learning

Learning through imitation of human behavior is a nice way to speedup the learning.
In Coach, this can be done in two steps -

1. Create a dataset of demonstrations by playing with the environment as a human.
   After this step, a pickle of the replay buffer containing your game play will be stored in the experiment directory.
   The path to this replay buffer will be printed to the screen.
   To do so, you should select an environment type and level through the command line, and specify the `--play` flag.

    *Example:*

    `python coach.py -et Doom -lvl Basic --play`


2. Next, use an imitation learning preset and set the replay buffer path accordingly.
    The path can be set either from the command line or from the preset itself.

    *Example:*

    `python coach.py -p Doom_Basic_BC -cp='agent.load_memory_from_file_path=\"<experiment dir>/replay_buffer.p\"'`


## Visualizations

### Rendering the Environment

Rendering the environment can be done by using the `-r` flag.
When working with multi-threaded algorithms, the rendered image will be representing the game play of the evaluation worker.
When working with single-threaded algorithms, the rendered image will be representing the single worker which can be either training or evaluating.
Keep in mind that rendering the environment in single-threaded algorithms may slow the training to some extent.
When playing with the environment using the `--play` flag, the environment will be rendered automatically without the need for specifying the `-r` flag.

*Example:*

`python coach.py -p Breakout_DQN -r`

### Dumping GIFs

Coach allows storing GIFs of the agent game play.
To dump GIF files, use the `-dg` flag.
The files are dumped after every evaluation episode, and are saved into the experiment directory, under a gifs sub-directory.

*Example:*

`python coach.py -p Breakout_A3C -n 4 -dg`

## Switching between deep learning frameworks

Coach uses TensorFlow as its main backend framework, but it also supports neon for some of the algorithms.
By default, TensorFlow will be used. It is possible to switch to neon using the `-f` flag.

*Example:*

`python coach.py -p Doom_Basic_DQN -f neon`

## Additional Flags

There are several convenient flags which are important to know about.
Here we will list most of the flags, but these can be updated from time to time.
The most up to date description can be found by using the `-h` flag.


|Flag                           |Type      |Description   |
|-------------------------------|----------|--------------|
|`-p PRESET`, ``--preset PRESET`|string    |Name of a preset to run (as configured in presets.py)         |
|`-l`, `--list`                 |flag      |List all available presets|
|`-e EXPERIMENT_NAME`, `--experiment_name EXPERIMENT_NAME`|string|Experiment name to be used to store the results.|
|`-r`, `--render`               |flag      |Render environment|
|`-f FRAMEWORK`, `--framework FRAMEWORK`|string|Neural network framework. Available values: tensorflow, neon|
|`-n NUM_WORKERS`, `--num_workers NUM_WORKERS`|int|Number of workers for multi-process based agents, e.g. A3C|
|`--play`                       |flag      |Play as a human by controlling the game with the keyboard. This option will save a replay buffer with the game play.|
|`--evaluate`                   |flag      |Run evaluation only. This is a convenient way to disable training in order to evaluate an existing checkpoint.|
|`-v`, `--verbose`              |flag      |Don't suppress TensorFlow debug prints.|
|`-s SAVE_MODEL_SEC`, `--save_model_sec SAVE_MODEL_SEC`|int|Time in seconds between saving checkpoints of the model.|
|`-crd CHECKPOINT_RESTORE_DIR`, `--checkpoint_restore_dir CHECKPOINT_RESTORE_DIR`|string|Path to a folder containing a checkpoint to restore the model from.|
|`-dg`, `--dump_gifs`           |flag      |Enable the gif saving functionality.|
|`-at AGENT_TYPE`, `--agent_type AGENT_TYPE`|string|Choose an agent type class to override on top of the selected preset. If no preset is defined, a preset can be set from the command-line by combining settings which are set by using `--agent_type`, `--experiment_type`, `--environemnt_type`|
|`-et ENVIRONMENT_TYPE`, `--environment_type ENVIRONMENT_TYPE`|string|Choose an environment type class to override on top of the selected preset. If no preset is defined, a preset can be set from the command-line by combining settings which are set by using `--agent_type`, `--experiment_type`, `--environemnt_type`|
|`-ept EXPLORATION_POLICY_TYPE`, `--exploration_policy_type EXPLORATION_POLICY_TYPE`|string|Choose an exploration policy type class to override on top of the selected preset.If no preset is defined, a preset can be set from the command-line by combining settings which are set by using `--agent_type`, `--experiment_type`, `--environemnt_type`|
|`-lvl LEVEL`, `--level LEVEL`  |string|Choose the level that will be played in the environment that was selected. This value will override the level parameter in the environment class.|
|`-cp CUSTOM_PARAMETER`, `--custom_parameter CUSTOM_PARAMETER`|string| Semicolon separated parameters used to override specific parameters on top of the selected preset (or on top of the command-line assembled one). Whenever a parameter value is a string, it should be inputted as `'\"string\"'`. For ex.: `"visualization.render=False;` `num_training_iterations=500;` `optimizer='rmsprop'"`|