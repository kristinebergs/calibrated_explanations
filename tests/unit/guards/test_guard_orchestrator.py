import numpy as np

from calibrated_explanations.guards.orchestrator import GuardOrchestrator


class DummyIntervalLearner:
    """Mock interval learner supporting uq_interval."""
    def predict(self, x, uq_interval=False):
        n_samples = len(x)
        preds = np.ones(n_samples) * 0.5
        if uq_interval:
            lower = np.zeros(n_samples)
            upper = np.ones(n_samples)
            return preds, (lower, upper)
        # Legacy format for backward compatibility
        return [(0.0, 1.0) for _ in range(n_samples)]


class FakeExplainer:
    def __init__(self, x_cal, y_cal):
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.interval_learner = DummyIntervalLearner()
        self.num_features = x_cal.shape[1]


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
    prediction = {
        "predict": np.array([0.0, 1.0]),
        "low": np.array([0.0, 0.0]),
        "high": np.array([1.0, 1.0]),
    }

    fx, ff = orch.filter_perturbations(perturbed_x, perturbed_feature, x_cal, prediction)
    assert fx.shape[0] == 1
    assert ff.shape[0] == 1

