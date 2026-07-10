"""Parity checks for invariant failures across validator and bridge entry points."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import LegacyPredictBridge
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    validate_explanation_batch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.utils.exceptions import ValidationError


class DummyContainer(CalibratedExplanations):
    """Concrete container type for protocol validation tests."""


class DummyExplanation(CalibratedExplanation):
    """Concrete explanation type for protocol validation tests."""


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"predict": np.array([0.5]), "low": np.array([0.6]), "high": np.array([0.4])},
            "Interval invariant violated",
        ),
        (
            {"predict": np.array([0.9]), "low": np.array([0.4]), "high": np.array([0.6])},
            "Prediction invariant violated",
        ),
    ],
)
def test_should_raise_validationerror_with_consistent_shape_when_invariants_fail(
    payload: dict[str, np.ndarray],
    expected: str,
) -> None:
    """Equivalent invalid payloads should fail consistently in both entry points."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": payload}],
        collection_metadata={"task": "regression", "mode": "test"},
    )

    with pytest.raises(ValidationError, match=expected):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="test")

    mock_explainer = Mock()
    mock_explainer.predict.return_value = (
        payload["predict"],
        (payload["low"], payload["high"]),
    )
    bridge = LegacyPredictBridge(mock_explainer)

    with pytest.raises(ValidationError, match=expected):
        bridge.predict("X", mode="regression", task="regression")


# ---------------------------------------------------------------------------
# ADR-015 gap 3 — task-scoped enforcement parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_should_enforce_interval_invariant_consistently_across_task_types(task):
    """Interval invariant violations must raise ValidationError for both regression and classification."""
    invalid_payload = {"predict": np.array([0.5]), "low": np.array([0.6]), "high": np.array([0.4])}
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": invalid_payload}],
        collection_metadata={"task": task, "mode": "test"},
    )
    with pytest.raises(ValidationError, match="Interval invariant violated"):
        validate_explanation_batch(batch, expected_task=task, expected_mode="test")


def test_should_enforce_prediction_invariant_for_regression_batches():
    """Regression batch predictions must remain inside their numeric intervals."""
    invalid_payload = {"predict": np.array([0.9]), "low": np.array([0.4]), "high": np.array([0.6])}
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": invalid_payload}],
        collection_metadata={"task": "regression", "mode": "test"},
    )
    with pytest.raises(ValidationError, match="Prediction invariant violated"):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="test")


def test_should_allow_classification_batch_scores_outside_interval_when_payload_represents_labels():
    """Classification payloads may combine calibrated score intervals with class labels."""
    payload = {
        "predict": np.array([0.9]),
        "low": np.array([0.4]),
        "high": np.array([0.6]),
        "classes": np.array([1]),
    }
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": payload}],
        collection_metadata={"task": "classification", "mode": "test"},
    )

    validated = validate_explanation_batch(
        batch, expected_task="classification", expected_mode="test"
    )

    assert validated is batch


def test_should_allow_classification_alternative_reference_prediction_when_task_uses_label_payload():
    """Alternative classification reference predictions should reuse the relaxed invariant."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[
            {
                "mode": "alternative",
                "reference_prediction": {
                    "predict": np.array([0.9]),
                    "low": np.array([0.4]),
                    "high": np.array([0.6]),
                    "classes": np.array([1]),
                },
                "rules": [
                    {
                        "predicted_value": 0.3,
                        "prediction_interval": {"low": 0.2, "high": 0.4},
                    }
                ],
            }
        ],
        collection_metadata={"task": "classification", "mode": "alternative"},
    )

    validated = validate_explanation_batch(
        batch, expected_task="classification", expected_mode="alternative"
    )

    assert validated is batch


def test_should_allow_classification_bridge_class_predictions_outside_score_interval():
    """Classification bridge should tolerate class labels with probability intervals."""
    mock_explainer = Mock()
    mock_explainer.predict.return_value = (
        np.asarray([0.9]),
        (np.asarray([0.4]), np.asarray([0.6])),
    )
    mock_explainer.predict.side_effect = [
        (
            np.asarray([0.9]),
            (np.asarray([0.4]), np.asarray([0.6])),
        ),
        np.asarray([1]),
    ]
    bridge = LegacyPredictBridge(mock_explainer)

    payload = bridge.predict("X", mode="factual", task="classification")

    np.testing.assert_allclose(payload["predict"], [0.9])
    np.testing.assert_allclose(payload["low"], [0.4])
    np.testing.assert_allclose(payload["high"], [0.6])
    np.testing.assert_array_equal(payload["classes"], [1])


def test_should_validate_rule_level_weights_when_batch_contains_factual_rules():
    """Factual rule weights must stay inside their calibrated intervals."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[
            {
                "rules": [
                    {
                        "weight": 0.9,
                        "weight_interval": {"low": 0.2, "high": 0.4},
                    }
                ]
            }
        ],
        collection_metadata={"task": "regression", "mode": "factual"},
    )

    with pytest.raises(ValidationError, match="rule 0 weight"):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="factual")


@pytest.mark.parametrize(
    "rule",
    [
        {"weight": 0.3, "weight_interval": {"low": 0.2, "high": 0.4}},
        {"weight": np.float64(0.3), "weight_interval": {"low": 0.3, "high": 0.3}},
    ],
)
def test_should_accept_rule_level_weight_inside_interval(rule):
    """Factual rule weights inside calibrated intervals must pass validation."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"rules": [rule]}],
        collection_metadata={"task": "regression", "mode": "factual"},
    )

    validated = validate_explanation_batch(
        batch, expected_task="regression", expected_mode="factual"
    )

    assert validated is batch


def test_should_validate_predicted_value_when_batch_contains_alternative_rules():
    """Alternative rule predicted values must stay inside their prediction intervals."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[
            {
                "reference_prediction": {"predict": 0.5, "low": 0.4, "high": 0.6},
                "rules": [
                    {
                        "predicted_value": 0.9,
                        "prediction_interval": {"low": 0.2, "high": 0.4},
                    }
                ],
            }
        ],
        collection_metadata={"task": "regression", "mode": "alternative"},
    )

    with pytest.raises(ValidationError, match="rule 0 predicted_value"):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="alternative")


def test_should_require_predicted_value_when_batch_contains_alternative_rule():
    """Alternative rules must include the scenario prediction they explain."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[
            {
                "reference_prediction": {"predict": 0.5, "low": 0.4, "high": 0.6},
                "rules": [{"prediction_interval": {"low": 0.2, "high": 0.4}}],
            }
        ],
        collection_metadata={"task": "regression", "mode": "alternative"},
    )

    with pytest.raises(ValidationError, match="requires predicted_value"):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="alternative")
