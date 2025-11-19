import numpy as np

from calibrated_explanations.core.explain.guard_orchestrator import GuardOrchestratorPlugin
from calibrated_explanations.plugins.guards import GuardContext


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
    
    # Create GuardContext for initialization
    context = GuardContext(
        task="classification",
        mode="factual",
        learner=None,
        x_cal=x_cal,
        y_cal=y_cal,
        interval_learner=expl.interval_learner,
        feature_names=["f0", "f1"],
        categorical_features=[],
        num_features=2,
        metadata={},
    )
    
    orch = GuardOrchestratorPlugin()
    orch.initialize(context)

    # Create dummy guard that rejects second perturbed row
    class DummyGuard:
        """Dummy guard for testing."""
        def __init__(self):
            self._fitted = True
        
        def accept_batch(self, arr, preds=None):  # noqa: ARG002
            """Accept batch method."""
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

    filtered_x, filtered_feature = orch.filter_perturbations(
        perturbed_x, perturbed_feature, x_cal, prediction
    )
    assert filtered_x.shape[0] == 1
    assert filtered_feature.shape[0] == 1

