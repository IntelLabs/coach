A custom observation filter implementation should look like this:

```bash
from coach.filters.filter import ObservationFilter

class CustomFilter(ObservationFilter):
  def __init__(self):
    ...
  def filter(self, env_response: EnvResponse) -> EnvResponse:
    ...
  def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
    ...
  def validate_input_observation_space(self, input_observation_space: ObservationSpace):
    ...
  def reset(self):
    ...
```

or for reward filters:
```bash
from coach.filters.filter import RewardFilter

class CustomFilter(ObservationFilter):
  def __init__(self):
    ...
  def filter(self, env_response: EnvResponse) -> EnvResponse:
    ...
  def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
    ...
  def reset(self):
    ...
```

To create a stack of filters:

```bash
from coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter, RescaleInterpolationType
from coach.filters.observation.observation_crop_filter import ObservationCropFilter
from coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from environments.environment_interface import ObservationSpace
import numpy as np
from core_types import EnvResponse
from filters.filter import InputFilter
from collections import OrderedDict

env_response = EnvResponse({'observation': np.ones([210, 160])}, reward=100, game_over=False)

rescale = ObservationRescaleToSizeFilter(
    output_observation_space=ObservationSpace(np.array([110, 84])),
    rescaling_interpolation_type=RescaleInterpolationType.BILINEAR
)

crop = ObservationCropFilter(
    crop_low=np.array([16, 0]),
    crop_high=np.array([100, 84])
)

clip = RewardClippingFilter(
    clipping_low=-1,
    clipping_high=1
)

input_filter = InputFilter(
    observation_filters=OrderedDict([('rescale', rescale), ('crop', crop)]),
    reward_filters=OrderedDict([('clip', clip)])
)

result = input_filter.filter(env_response)

```