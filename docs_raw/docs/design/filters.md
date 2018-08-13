# Filters

Filters are a mechanism in Coach that allows doing pre-processing and post-processing of the internal agent information.
There are two filter categories -

* **Input filters** - these are filters that process the information passed **into** the agent from the environment.
  This information includes the observation and the reward. Input filters therefore allow rescaling observations,
  normalizing rewards, stack observations, etc.

* **Output filters** - these are filters that process the information going **out** of the agent into the environment.
  This information includes the action the agent chooses to take. Output filters therefore allow conversion of
  actions from one space into another. For example, the agent can take $ N $ discrete actions, that will be mapped by
  the output filter onto $ N $ continuous actions.

Filters can be stacked on top of each other in order to build complex processing flows of the inputs or outputs.

<p style="text-align: center;">

<img src="../../img/filters.png" alt="Filters mechanism" style="width: 350px;"/>

</p>

## Input Filters

The input filters are separated into two categories - **observation filters** and **reward filters**.

### Observation Filters

* **ObservationClippingFilter** - Clips the observation values to a given range of values. For example, if the
  observation consists of measurements in an arbitrary range, and we want to control the minimum and maximum values
  of these observations, we can define a range and clip the values of the measurements.

* **ObservationCropFilter** - Crops the size of the observation to a given crop window. For example, in Atari, the
  observations are images with a shape of 210x160. Usually, we will want to crop the size of the observation to a
  square of 160x160 before rescaling them.

* **ObservationMoveAxisFilter** - Reorders the axes of the observation. This can be useful when the observation is an
  image, and we want to move the channel axis to be the last axis instead of the first axis.

* **ObservationNormalizationFilter** - Normalizes the observation values with a running mean and standard deviation of
  all the observations seen so far. The normalization is performed element-wise. Additionally, when working with
  multiple workers, the statistics used for the normalization operation are accumulated over all the workers.

* **ObservationReductionBySubPartsNameFilter** - Allows keeping only parts of the observation, by specifying their
  name. For example, the CARLA environment extracts multiple measurements that can be used by the agent, such as
  speed and location. If we want to only use the speed, it can be done using this filter.

* **ObservationRescaleSizeByFactorFilter** - Rescales an image observation by some factor. For example, the image size
  can be reduced by a factor of 2.

* **ObservationRescaleToSizeFilter** - Rescales an image observation to a given size. The target size does not
  necessarily keep the aspect ratio of the original observation.

* **ObservationRGBToYFilter** - Converts a color image observation specified using the RGB encoding into a grayscale
  image observation, by keeping only the luminance (Y) channel of the YUV encoding. This can be useful if the colors
  in the original image are not relevant for solving the task at hand.

* **ObservationSqueezeFilter** - Removes redundant axes from the observation, which are axes with a dimension of 1.

* **ObservationStackingFilter** - Stacks several observations on top of each other. For image observation this will
  create a 3D blob. The stacking is done in a lazy manner in order to reduce memory consumption. To achieve this,
  a LazyStack object is used in order to wrap the observations in the stack. For this reason, the
  ObservationStackingFilter **must** be the last filter in the inputs filters stack.

* **ObservationUint8Filter** - Converts a floating point observation into an unsigned int 8 bit observation. This is
  mostly useful for reducing memory consumption and is usually used for image observations. The filter will first
  spread the observation values over the range 0-255 and then discretize them into integer values.

### Reward Filters

* **RewardClippingFilter** - Clips the reward values into a given range. For example, in DQN, the Atari rewards are
  clipped into the range -1 and 1 in order to control the scale of the returns.

* **RewardNormalizationFilter** -  Normalizes the reward values with a running mean and standard deviation of
  all the rewards seen so far. When working with multiple workers, the statistics used for the normalization operation
  are accumulated over all the workers.

* **RewardRescaleFilter** - Rescales the reward by a given factor. Rescaling the rewards of the environment has been
  observed to have a large effect (negative or positive) on the behavior of the learning process.

## Output Filters

The output filters only process the actions.

### Action Filters

* **AttentionDiscretization** - Discretizes an **AttentionActionSpace**. The attention action space defines the actions
  as choosing sub-boxes in a given box. For example, consider an image of size 100x100, where the action is choosing
  a crop window of size 20x20 to attend to in the image. AttentionDiscretization allows discretizing the possible crop
  windows to choose into a finite number of options, and map a discrete action space into those crop windows.

* **BoxDiscretization** - Discretizes a continuous action space into a discrete action space, allowing the usage of
  agents such as DQN for continuous environments such as MuJoCo. Given the number of bins to discretize into, the
  original continuous action space is uniformly separated into the given number of bins, each mapped to a discrete
  action index. For example, if the original actions space is between -1 and 1 and 5 bins were selected, the new action
  space will consist of 5 actions mapped to -1, -0.5, 0, 0.5 and 1.

* **BoxMasking** - Masks part of the action space to enforce the agent to work in a defined space. For example,
  if the original action space is between -1 and 1, then this filter can be used in order to constrain the agent actions
  to the range 0 and 1 instead. This essentially masks the range -1 and 0 from the agent.

* **PartialDiscreteActionSpaceMap** - Partial map of two countable action spaces. For example, consider an environment
  with a MultiSelect action space (select multiple actions at the same time, such as jump and go right), with 8 actual
  MultiSelect actions. If we want the agent to be able to select only 5 of those actions by their index (0-4), we can
  map a discrete action space with 5 actions into the 5 selected MultiSelect actions. This will both allow the agent to
  use regular discrete actions, and mask 3 of the actions from the agent.

* **FullDiscreteActionSpaceMap** - Full map of two countable action spaces. This works in a similar way to the
  PartialDiscreteActionSpaceMap, but maps the entire source action space into the entire target action space, without
  masking any actions.

* **LinearBoxToBoxMap** - A linear mapping of two box action spaces. For example, if the action space of the
  environment consists of continuous actions between 0 and 1, and we want the agent to choose actions between -1 and 1,
  the LinearBoxToBoxMap can be used to map the range -1 and 1 to the range 0 and 1 in a linear way. This means that the
  action -1 will be mapped to 0, the action 1 will be mapped to 1, and the rest of the actions will be linearly mapped
  between those values.
