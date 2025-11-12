import numpy as np


class DummyGuard:
    def __init__(self, accept_ret=True):
        self._accept_ret = accept_ret

    def accept(self, x_conj, label_ctx):
        return self._accept_ret


def test_validate_conjunction_calls_guard():
    from calibrated_explanations.guards.conjunctions import validate_conjunction

    x = np.array([0.1, 0.2])
    g_true = DummyGuard(accept_ret=True)
    g_false = DummyGuard(accept_ret=False)

    assert validate_conjunction(x, g_true, 0) is True
    assert validate_conjunction(x, g_false, 1) is False
