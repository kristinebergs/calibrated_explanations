"""Unit tests for the martingale e-test and EMartingale helper.

These tests exercise the small e-test helper used by perturbation guards and
the tiny EMartingale accumulator. Docstrings are present to satisfy
pydocstyle checks in the test suite.
"""

import numpy as np


def test_martingale_evalue_and_reject():
    """Ensure e_value raises when the test is not fitted and behaves for inliers.

    This test verifies that calling ``e_value`` before ``fit`` raises a
    ValueError, that a fitted test returns a non-negative float e-value, and
    that in-distribution points are not rejected when the gamma threshold is
    large.
    """
    from calibrated_explanations.guards.martingale import MartingaleETest

    # Create simple training data: 2D points on unit grid
    rng = np.random.default_rng(0)
    x_train = rng.random((100, 2))

    m = MartingaleETest(k=3, n_neighbors=10, gamma=10.0)
    # e_value should raise if not fitted
    try:
        m.e_value(np.array([0.0, 0.0]))
        raised = False
    except ValueError:
        raised = True
    assert raised

    m.fit(x_train)

    # e_value should return a positive float
    val = m.e_value(np.array([0.1, 0.1]))
    assert isinstance(val, float)
    assert val >= 0.0

    # With default gamma large, reject should be False for inlier
    assert m.reject(np.array([0.1, 0.1])) is False


def test_emartingale_basic_updates():
    """Verify EMartingale updates, product semantics and reset behavior."""
    from calibrated_explanations.guards.martingale import EMartingale

    m = EMartingale()
    assert m.current_value() == 1.0
    assert m.n_updates == 0

    # update with two e-values and check product
    m.update(1.5)
    assert m.n_updates == 1
    v1 = m.current_value()
    assert isinstance(v1, float) and v1 > 0.0

    m.update(0.5)
    assert m.n_updates == 2
    # product should be 1.5 * 0.5 = 0.75
    assert abs(m.current_value() - 0.75) < 1e-12

    # reset should bring back to neutral
    m.reset()
    assert m.current_value() == 1.0
    assert m.n_updates == 0


def test_emartingale_with_martingale_test():
    """Check EMartingale can be updated from a MartingaleETest e-value."""
    from calibrated_explanations.guards.martingale import EMartingale, MartingaleETest

    rng = np.random.default_rng(1)
    x_train = rng.random((50, 2))
    mt = MartingaleETest(k=3, n_neighbors=10, gamma=10.0)
    mt.fit(x_train)

    em = EMartingale()
    # update_from_test should compute e-value and update internal product
    e = em.update_from_test(np.array([0.1, 0.1]), mt)
    assert isinstance(e, float)
    assert em.n_updates == 1
    # e must match current value since it was the first update
    assert abs(em.current_value() - e) < 1e-12


def test_martingale_rejects_far_point():
    """Assert that a very far point is rejected by the martingale e-test."""
    from calibrated_explanations.guards.martingale import MartingaleETest

    rng = np.random.default_rng(2)
    # Training data concentrated near the origin
    x_train = rng.normal(loc=0.0, scale=0.5, size=(200, 2))

    mt = MartingaleETest(k=5, n_neighbors=50, gamma=10.0)
    mt.fit(x_train)

    # An in-distribution point should not be rejected
    inlier = x_train[0]
    assert mt.reject(inlier) is False

    # A far-away outlier should be rejected (very large distance -> large e-value)
    outlier = np.array([1e3, 1e3])
    assert mt.reject(outlier) is True
