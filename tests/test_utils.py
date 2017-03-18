from giggle.utils import (
    grouper,
)


def test_grouper():
    xs = [1, 2, 3, 4]
    assert list(grouper(xs, 3, None)) == [(1, 2, 3), (4, None, None)]
    assert list(grouper(xs, 1, None)) == [(1, ), (2, ), (3, ), (4, )]
