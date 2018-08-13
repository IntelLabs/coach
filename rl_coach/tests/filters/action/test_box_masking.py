import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.filters.action.box_masking import BoxMasking
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace
import numpy as np


@pytest.mark.unit_test
def test_filter():
    filter = BoxMasking(10, 20)

    # passing an output space that is wrong
    with pytest.raises(ValueError):
        filter.validate_output_action_space(DiscreteActionSpace(10))

    # 1 dimensional box
    output_space = BoxActionSpace(1, 5, 30)
    input_space = filter.get_unfiltered_action_space(output_space)

    action = np.array([2])
    result = filter.filter(action)
    assert result == np.array([12])
    assert output_space.val_matches_space_definition(result)

