import warnings
from unittest.mock import create_autospec

import numpy as np
import pytest

from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor
from calibrated_explanations.plugins.predict import PredictBridge
from calibrated_explanations.utils.exceptions import ValidationError


class DummyBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.predictions = {
            "predict": {"result": "predict"},
            "predict_interval": ("interval",),
            "predict_proba": (0.1, 0.9),
        }

    def predict(self, x, *, mode, task, bins=None):
        self.calls.append(("predict", (mode, task, bins)))
        return self.predictions["predict"]

    def predict_interval(self, x, *, task, bins=None):
        self.calls.append(("predict_interval", (task, bins)))
        return self.predictions["predict_interval"]

    def predict_proba(self, x, bins=None):
        self.calls.append(("predict_proba", (bins,)))
        return self.predictions["predict_proba"]


def test_predict_bridge_monitor_tracks_usage_and_passthrough():
    """Test that PredictBridgeMonitor correctly tracks bridge method calls and passes results."""
    bridge = DummyBridge()
    monitor = PredictBridgeMonitor(bridge)

    assert monitor.used is False

    predict_result = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")
    interval_result = monitor.predict_interval(np.array([[1.0]]), task="classification")
    proba_result = monitor.predict_proba(np.array([[1.0]]))

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used is True

    # Ensure the wrapped bridge is called transparently.
    assert predict_result is bridge.predictions["predict"]
    assert interval_result is bridge.predictions["predict_interval"]
    assert proba_result is bridge.predictions["predict_proba"]

    assert bridge.calls[0][0] == "predict"
    assert bridge.calls[1][0] == "predict_interval"
    assert bridge.calls[2][0] == "predict_proba"


def test_predict_bridge_monitor_reset_usage():
    """Test that usage tracking can be reset."""
    bridge = create_autospec(PredictBridge, instance=True)
    monitor = PredictBridgeMonitor(bridge)

    payload = {"x": np.ones((2, 2))}
    monitor.predict(payload, mode="factual", task="classification")

    assert monitor.used
    assert len(monitor.calls) > 0

    monitor.reset_usage()
    assert monitor.calls == ()
    assert not monitor.used


def test_should_raise_when_predict_monitor_interval_tuple_violates_bounds():
    class IntervalBridge(DummyBridge):
        def predict_interval(self, x, *, task, bins=None):
            return (
                np.asarray([0.4]),
                np.asarray([0.5]),
                np.asarray([0.3]),
            )

    monitor = PredictBridgeMonitor(IntervalBridge())
    with pytest.raises(ValidationError, match="low > high"):
        monitor.predict_interval(np.array([[1.0]]), task="classification")


def test_should_raise_when_predict_monitor_regression_prediction_outside_interval():
    class BadPredictBridge(DummyBridge):
        def __init__(self) -> None:
            super().__init__()
            self.predictions["predict"] = {
                "predict": np.asarray([0.9]),
                "low": np.asarray([0.4]),
                "high": np.asarray([0.6]),
            }

    monitor = PredictBridgeMonitor(BadPredictBridge())

    with pytest.raises(ValidationError, match="predict not in \\[low, high\\]"):
        monitor.predict(np.array([[1.0]]), mode="factual", task="regression")


def test_should_allow_predict_monitor_classification_score_interval_payload():
    class ClassificationBridge(DummyBridge):
        def __init__(self) -> None:
            super().__init__()
            self.predictions["predict"] = {
                "predict": np.asarray([0.9]),
                "low": np.asarray([0.4]),
                "high": np.asarray([0.6]),
                "classes": np.asarray([1]),
            }

    monitor = PredictBridgeMonitor(ClassificationBridge())

    payload = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")

    assert payload["classes"][0] == 1


def test_should_return_when_predict_monitor_payload_has_empty_arrays():
    class EmptyBridge(DummyBridge):
        def __init__(self) -> None:
            super().__init__()
            self.predictions["predict"] = {
                "predict": np.asarray([]),
                "low": np.asarray([]),
                "high": np.asarray([]),
            }

    monitor = PredictBridgeMonitor(EmptyBridge())

    payload = monitor.predict(np.array([[1.0]]), mode="factual", task="regression")

    assert payload["predict"].size == 0


def test_should_return_when_predict_monitor_payload_is_non_numeric():
    class NonNumericBridge(DummyBridge):
        def __init__(self) -> None:
            super().__init__()
            self.predictions["predict"] = {
                "predict": np.asarray(["class-a"], dtype=object),
                "low": np.asarray(["low"], dtype=object),
                "high": np.asarray(["high"], dtype=object),
            }

    monitor = PredictBridgeMonitor(NonNumericBridge())

    payload = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")

    assert payload["predict"][0] == "class-a"


def test_should_raise_when_predict_monitor_regression_interval_tuple_prediction_outside_bounds():
    class RegressionIntervalBridge(DummyBridge):
        def predict_interval(self, x, *, task, bins=None):
            return (
                np.asarray([0.9]),
                np.asarray([0.4]),
                np.asarray([0.6]),
            )

    monitor = PredictBridgeMonitor(RegressionIntervalBridge())

    with pytest.raises(ValidationError, match="predict not in \\[low, high\\]"):
        monitor.predict_interval(np.array([[1.0]]), task="regression")


def test_predict_monitor_handles_conversion_errors(monkeypatch: pytest.MonkeyPatch):
    class StrangeBridge(DummyBridge):
        def __init__(self) -> None:
            super().__init__()
            self.predictions["predict"] = {"predict": object(), "low": object(), "high": object()}

    monitor = PredictBridgeMonitor(StrangeBridge())

    def explode(*args, **kwargs):
        raise TypeError("bad")

    monkeypatch.setattr(np, "asanyarray", explode)
    with warnings.catch_warnings(record=True) as captured:
        monitor.predict(np.array([[1.0]]), mode="factual", task="regression")
    assert captured == []
