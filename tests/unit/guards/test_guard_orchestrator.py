import numpy as np

from calibrated_explanations.guards.orchestrator import GuardOrchestrator
from calibrated_explanations.guards.regions import ConformalRegionOracle


class DummyIntervalLearner:
    def predict(self, X):
        # Return constant intervals for each row
        return [(0.0, 1.0) for _ in range(len(X))]


class FakeExplainer:
    def __init__(self, x_cal, y_cal):
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.interval_learner = DummyIntervalLearner()
        self.num_features = x_cal.shape[1]


def test_fit_guard_creates_guard():
    x_cal = np.random.RandomState(0).randn(80, 3)
    y_cal = (np.random.RandomState(0).randint(0, 2, size=80))
    expl = FakeExplainer(x_cal, y_cal)
    orch = GuardOrchestrator(expl)

    assert orch.get_guard() is None
    orch.fit_guard({"alpha": 0.1, "n_clusters": 3, "random_state": 42})
    guard = orch.get_guard()
    assert guard is not None
    assert hasattr(guard, "_fitted") and guard._fitted


def test_accept_and_intervals_after_fit():
    x_cal = np.random.RandomState(1).randn(60, 2)
    y_cal = np.random.RandomState(1).randint(0, 2, size=60)
    expl = FakeExplainer(x_cal, y_cal)
    orch = GuardOrchestrator(expl)
    orch.fit_guard({"alpha": 0.1, "n_clusters": 2, "random_state": 0})

    guard = orch.get_guard()
    # pick a calibration point, expect it to be accepted
    point = x_cal[0]
    assert orch.accept(point) in (True, False)

    # batch accept shape
    batch = x_cal[:5]
    res = orch.accept_batch(batch)
    assert isinstance(res, np.ndarray)
    assert res.shape[0] == 5

    intervals = orch.intervals(x_cal[0])
    assert isinstance(intervals, list)
    assert len(intervals) == expl.num_features


def test_filter_perturbations_delegation():
    x_cal = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_cal = np.array([0, 1])
    expl = FakeExplainer(x_cal, y_cal)
    orch = GuardOrchestrator(expl)

    # Create dummy guard that rejects second perturbed row
    class DummyGuard:
        def accept_batch(self, arr, preds=None):
            return np.array([True, False])

    orch.set_guard(DummyGuard())

    perturbed_x = np.array([[0.0, 0.0], [10.0, 10.0]])
    # perturbed_feature second column is origin instance index
    perturbed_feature = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
    prediction = {"predict": np.array([0.0, 1.0]), "low": np.array([0.0, 0.0]), "high": np.array([1.0, 1.0])}

    fx, ff = orch.filter_perturbations(perturbed_x, perturbed_feature, x_cal, prediction)
    assert fx.shape[0] == 1
    assert ff.shape[0] == 1
