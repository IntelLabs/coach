import pytest

from rl_coach.saver import Saver, SaverCollection


@pytest.mark.unit_test
def test_checkpoint_collection():
    class SaverTest(Saver):
        def __init__(self, path):
            self._path = path
            self._count = 1

        @property
        def path(self):
            return self._path

        def merge(self, other: 'Saver'):
            assert isinstance(other, SaverTest)
            assert self.path == other.path
            self._count += other._count

    # test add
    savers = SaverCollection(SaverTest('123'))
    savers.add(SaverTest('123'))
    savers.add(SaverTest('456'))

    def check_collection(mul):
        paths = ['123', '456']
        for c in savers:
            paths.remove(c.path)
            if c.path == '123':
                assert c._count == 2 * mul
            elif c.path == '456':
                assert c._count == 1 * mul
            else:
                assert False, "invalid path"

    check_collection(1)

    # test update
    savers.update(savers)
    check_collection(2)
