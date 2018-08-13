import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from rl_coach.filters.action.box_discretization import BoxDiscretization
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace


@pytest.mark.unit_test
def test_filter():
    filter = BoxDiscretization(9)

    # passing an output space that is wrong
    with pytest.raises(ValueError):
        filter.validate_output_action_space(DiscreteActionSpace(10))

    # 1 dimensional box
    output_space = BoxActionSpace(1, 5, 15)
    input_space = filter.get_unfiltered_action_space(output_space)

    assert filter.target_actions == [[5.], [6.25], [7.5], [8.75], [10.], [11.25], [12.5], [13.75], [15.]]
    assert input_space.actions == list(range(9))

    action = 2

    result = filter.filter(action)
    assert result == [7.5]
    assert output_space.val_matches_space_definition(result)

    # 2 dimensional box
    filter = BoxDiscretization(3)
    output_space = BoxActionSpace(2, 5, 15)
    input_space = filter.get_unfiltered_action_space(output_space)

    assert filter.target_actions == [[5., 5.], [5., 10.], [5., 15.],
                                     [10., 5.], [10., 10.], [10., 15.],
                                     [15., 5.], [15., 10.], [15., 15.]]
    assert input_space.actions == list(range(9))

    action = 2

    result = filter.filter(action)
    assert result == [5., 15.]
    assert output_space.val_matches_space_definition(result)
