import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.filters.action.attention_discretization import AttentionDiscretization
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace, AttentionActionSpace
import numpy as np


@pytest.mark.unit_test
def test_filter():
    filter = AttentionDiscretization(2)

    # passing an output space that is wrong
    with pytest.raises(ValueError):
        filter.validate_output_action_space(DiscreteActionSpace(10))
    with pytest.raises(ValueError):
        filter.validate_output_action_space(BoxActionSpace(10))

    # 1 dimensional box
    output_space = AttentionActionSpace(2, 0, 83)
    input_space = filter.get_unfiltered_action_space(output_space)

    assert np.all(filter.target_actions == np.array([[[0., 0.], [41.5, 41.5]],
                                     [[0., 41.5], [41.5, 83.]],
                                     [[41.5, 0], [83., 41.5]],
                                     [[41.5, 41.5], [83., 83.]]]))
    assert input_space.actions == list(range(4))

    action = 2

    result = filter.filter(action)
    assert np.all(result == np.array([[41.5, 0], [83., 41.5]]))
    assert output_space.val_matches_space_definition(result)

    # force int bins
    filter = AttentionDiscretization(2, force_int_bins=True)
    input_space = filter.get_unfiltered_action_space(output_space)

    assert np.all(filter.target_actions == np.array([[[0., 0.], [41, 41]],
                                                     [[0., 41], [41, 83.]],
                                                     [[41, 0], [83., 41]],
                                                     [[41, 41], [83., 83.]]]))
