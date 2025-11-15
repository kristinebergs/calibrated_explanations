import numpy as np

from calibrated_explanations.guards.orchestrator import GuardOrchestrator


class DummyIntervalLearner:
    def predict(self, x):
        # return a fixed narrow interval for all instances
        return [(0.0, 0.1) for _ in range(len(x))]


class FakeExplainer:
    def __init__(self, x_cal, y_cal):
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.interval_learner = DummyIntervalLearner()
        self.num_features = x_cal.shape[1]


def test_guard_rejects_faraway_perturbations():
    # Build clustered training data near the origin
    rng = np.random.RandomState(0)
    x_train = rng.normal(loc=0.0, scale=0.5, size=(500, 3))
    y_train = rng.randint(0, 2, size=500)

    expl = FakeExplainer(x_train, y_train)
    orch = GuardOrchestrator(expl)

    # Fit a guard with moderate alpha so nearby perturbations are accepted
    # but far-away perturbations are rejected.
    orch.fit_guard({"alpha": 0.1, "n_clusters": 3, "random_state": 0})
    guard = orch.get_guard()

    # Two perturbations: one near origin (should be accepted), one far away (rejected)
    perturbed = np.array([[0.1, -0.05, 0.0], [100.0, 100.0, 100.0]])

    accepts = guard.accept_batch(perturbed)
    assert accepts.shape[0] == 2
    assert bool(accepts[0]) is True
    assert bool(accepts[1]) is False
