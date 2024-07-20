import pytest

__author__ = "awerdich"
__copyright__ = "awerdich"
__license__ = "CC"


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4
