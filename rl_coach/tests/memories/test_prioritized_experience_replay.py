# nasty hack to deal with issue #46
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from rl_coach.memories.non_episodic.prioritized_experience_replay import SegmentTree


@pytest.mark.unit_test
def test_sum_tree():
    # test power of 2 sum tree
    sum_tree = SegmentTree(size=4, operation=SegmentTree.Operation.SUM)
    sum_tree.add(10, "10")
    assert sum_tree.total_value() == 10
    sum_tree.add(20, "20")
    assert sum_tree.total_value() == 30
    sum_tree.add(5, "5")
    assert sum_tree.total_value() == 35
    sum_tree.add(7.5, "7.5")
    assert sum_tree.total_value() == 42.5
    sum_tree.add(2.5, "2.5")
    assert sum_tree.total_value() == 35
    sum_tree.add(5, "5")
    assert sum_tree.total_value() == 20

    assert sum_tree.get_element_by_partial_sum(2) == (0, 2.5, '2.5')
    assert sum_tree.get_element_by_partial_sum(3) == (1, 5.0, '5')
    assert sum_tree.get_element_by_partial_sum(10) == (2, 5.0, '5')
    assert sum_tree.get_element_by_partial_sum(13) == (3, 7.5, '7.5')

    sum_tree.update(2, 10)
    assert sum_tree.__str__() == "[25.]\n[ 7.5 17.5]\n[ 2.5  5.  10.   7.5]\n"

    # test non power of 2 sum tree
    with pytest.raises(ValueError):
        sum_tree = SegmentTree(size=5, operation=SegmentTree.Operation.SUM)


@pytest.mark.unit_test
def test_min_tree():
    min_tree = SegmentTree(size=4, operation=SegmentTree.Operation.MIN)
    min_tree.add(10, "10")
    assert min_tree.total_value() == 10
    min_tree.add(20, "20")
    assert min_tree.total_value() == 10
    min_tree.add(5, "5")
    assert min_tree.total_value() == 5
    min_tree.add(7.5, "7.5")
    assert min_tree.total_value() == 5
    min_tree.add(2, "2")
    assert min_tree.total_value() == 2
    min_tree.add(3, "3")
    min_tree.add(3, "3")
    min_tree.add(3, "3")
    min_tree.add(5, "5")
    assert min_tree.total_value() == 3


@pytest.mark.unit_test
def test_max_tree():
    max_tree = SegmentTree(size=4, operation=SegmentTree.Operation.MAX)
    max_tree.add(10, "10")
    assert max_tree.total_value() == 10
    max_tree.add(20, "20")
    assert max_tree.total_value() == 20
    max_tree.add(5, "5")
    assert max_tree.total_value() == 20
    max_tree.add(7.5, "7.5")
    assert max_tree.total_value() == 20
    max_tree.add(2, "2")
    assert max_tree.total_value() == 20
    max_tree.add(3, "3")
    max_tree.add(3, "3")
    max_tree.add(3, "3")
    max_tree.add(5, "5")
    assert max_tree.total_value() == 5

    # update
    max_tree.update(1, 10)
    assert max_tree.total_value() == 10
    assert max_tree.__str__() == "[10.]\n[10.  3.]\n[ 5. 10.  3.  3.]\n"
    max_tree.update(1, 2)
    assert max_tree.total_value() == 5
    assert max_tree.__str__() == "[5.]\n[5. 3.]\n[5. 2. 3. 3.]\n"


if __name__ == "__main__":
    test_sum_tree()
    test_min_tree()
    test_max_tree()
