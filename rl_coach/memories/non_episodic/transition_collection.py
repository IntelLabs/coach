from rl_coach.core_types import Transition


class TransitionCollection(object):
    """
    Simple python implementation of transitions collection non-episodic memories
    are constructed on top of.
    """
    def __init__(self):
        super(TransitionCollection, self).__init__()

    def append(self, transition):
        pass

    def extend(self, transitions):
        for transition in transitions:
            self.append(transition)

    def __len__(self):
        pass

    def __del__(self, range: slice):
        # NOTE: the only slice used is the form: slice(None, n)
        # NOTE: if it is easier, what we really want here is the ability to
        # constrain the size of the collection. as new transitions are added,
        # old transitions can be removed to maintain a maximum collection size.
        pass

    def __getitem__(self, key: int):
        # NOTE: we can switch to a method which fetches multiple items at a time
        # if that would significantly improve performance
        pass

    def __iter__(self):
        # this is not high priority
        pass
