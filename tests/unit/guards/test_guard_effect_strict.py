import numpy as np

from calibrated_explanations.guards.orchestrator import GuardOrchestrator


class DummyIntervalLearner:
    """Mock interval learner supporting uq_interval."""
    def predict(self, x, uq_interval=False):  # pylint: disable=missing-function-docstring
        # return a fixed narrow interval for all instances
        n_samples = len(x)
        preds = np.ones(n_samples) * 0.05
        if uq_interval:
            lower = np.zeros(n_samples)
            upper = np.ones(n_samples) * 0.1
            return preds, (lower, upper)
        return [(0.0, 0.1) for _ in range(n_samples)]


class FakeExplainer:
    def __init__(self, x_cal, y_cal):
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.interval_learner = DummyIntervalLearner()
        self.num_features = x_cal.shape[1]

