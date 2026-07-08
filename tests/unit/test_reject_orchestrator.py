from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


class DummyIntervalLearner:
    def predict_proba(self, x, bins=None):
        proba = np.array([0.4, 0.6])
        return np.tile(proba, (len(x), 1))


class DummyRejectLearner:
    def __init__(self):
        self.seeds = []

    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
        self.seeds.append(seed)
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.ones((n_rows, n_cols), dtype=float)

    def predict_set(self, alphas, bins=None, confidence=0.95, smoothing=True, seed=None):
        self.seeds.append(seed)
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.ones((n_rows, n_cols), dtype=bool)


class DummyExplainer:
    def __init__(self):
        self.mode = "classification"
        self.y_cal = np.array([0, 1])
        self.seed = 123
        self.interval_learner = DummyIntervalLearner()
        self.reject_learner = DummyRejectLearner()

    def is_multiclass(self):
        return False


def test_should_raise_validation_error_when_calibration_set_is_invalid():
    explainer = DummyExplainer()
    orchestrator = RejectOrchestrator(explainer)

    with pytest.raises(ValidationError):
        orchestrator.initialize_reject_learner(calibration_set="invalid")


class DummyRejectLearnerMonotonic:
    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
        n = len(alphas)
        num_classes = 3
        out = np.full((n, num_classes), 0.2, dtype=float)
        for i in range(min(n, 9)):
            out[i, i % num_classes] = 0.99
        if n >= 10:
            out[9, 0] = 0.60
            out[9, 1] = 0.60
        return out


class DummyRejectLearnerAlwaysEmpty:
    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.zeros((n_rows, n_cols), dtype=float)


class PatchedRejectOrchestrator(RejectOrchestrator):
    """RejectOrchestrator subclass that short-circuits predict_reject_breakdown."""

    def __init__(self, explainer):
        super().__init__(explainer)
        self.last_confidence = None

    def predict_reject_breakdown(self, x, bins=None, confidence=0.95, threshold=None):
        self.last_confidence = float(confidence)
        n = len(x)
        return {
            "rejected": np.zeros(n, dtype=bool),
            "error_rate": 0.0,
            "reject_rate": 1.0 - float(confidence),
        }


def make_patched_orchestrator():
    explainer = DummyExplainer()
    return PatchedRejectOrchestrator(explainer)


def test_should_accept_reject_confidence_kwarg_when_passed_to_predict_reject():
    orch = make_patched_orchestrator()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    rejected, _err, reject_rate = orch.predict_reject(x, reject_confidence=0.80)
    assert rejected.shape == (2,)
    assert orch.last_confidence == pytest.approx(0.80)
    assert reject_rate == pytest.approx(0.20)


def test_should_raise_configuration_error_when_removed_confidence_alias_is_passed_to_predict_reject():
    orch = make_patched_orchestrator()

    with pytest.raises(ConfigurationError, match="reject_confidence"):
        orch.predict_reject(np.array([[1.0, 2.0]]), confidence=0.5)


@pytest.mark.parametrize("method_name", ["predict_reject", "apply_policy"])
def test_should_raise_configuration_error_when_unknown_reject_kwarg_is_passed(method_name):
    orch = make_patched_orchestrator()

    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        if method_name == "predict_reject":
            orch.predict_reject(np.array([[1.0, 2.0]]), rejct_confidence=0.5)
        else:
            orch.apply_policy(RejectPolicy.FLAG, np.array([[1.0, 2.0]]), rejct_confidence=0.5)


def test_should_apply_reject_confidence_when_passed_to_apply_policy():
    orch = make_patched_orchestrator()
    seen_confidences = []

    def custom_strategy(policy, x, **kwargs):
        confidence = float(kwargs["confidence"])
        seen_confidences.append(confidence)
        return RejectResult(
            prediction=None,
            explanation=None,
            rejected=np.zeros(len(x), dtype=bool),
            policy=policy,
            metadata={"reject_rate": 1.0 - confidence},
        )

    orch.register_strategy("test.capture", custom_strategy)
    result = orch.apply_policy(
        RejectPolicy.FLAG,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        reject_confidence=0.5,
        strategy="test.capture",
    )

    assert seen_confidences == [pytest.approx(0.5)]
    assert result.metadata["reject_rate"] == pytest.approx(0.5)
