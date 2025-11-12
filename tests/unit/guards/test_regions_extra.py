import numpy as np


def test_regions_fit_reg_requires_threshold():
    from calibrated_explanations.guards.regions import ConformalRegionOracle

    guard = ConformalRegionOracle(mode="reg")
    rng = np.random.default_rng(1)
    x = rng.random((10, 2))
    y = rng.random(10)
    try:
        guard.fit(x, y)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_intervals_unfitted_and_missing_label():
    from calibrated_explanations.guards.regions import ConformalRegionOracle

    guard = ConformalRegionOracle()
    x_inst = np.array([0.1, 0.1])
    # Not fitted -> intervals should raise
    try:
        guard.intervals(x_inst, 0)
        raised = False
    except ValueError:
        raised = True
    assert raised

    # Now mark fitted but with no clusters for label
    guard._fitted = True
    guard._clusters = {}
    intervals = guard.intervals(x_inst, 0)
    assert isinstance(intervals, list)
    assert len(intervals) == 2
    assert intervals[0] == [] and intervals[1] == []


def test_accept_manual_clusters_and_martingale():
    from calibrated_explanations.guards.regions import ConformalRegionOracle
    from sklearn.neighbors import KDTree

    guard = ConformalRegionOracle()
    # Manually construct a simple single-cluster environment
    centers = np.array([[0.0, 0.0]])
    variances = np.array([[1.0, 1.0]])
    radius = 10.0

    guard._clusters = {0: centers}
    guard._variances = {0: variances}
    guard._radii = {0: radius}
    guard._trees = {0: KDTree(centers)}
    guard._fitted = True

    # Inlier at center should be accepted
    assert guard.accept(np.array([0.0, 0.0]), 0) is True

    # Far away point should be rejected
    assert guard.accept(np.array([100.0, 100.0]), 0) is False

    # If martingale is present and rejects, accept should be False
    class FakeM:
        def reject(self, x):
            return True

    guard._martingale = FakeM()
    # Now even center will be rejected
    assert guard.accept(np.array([0.0, 0.0]), 0) is False
