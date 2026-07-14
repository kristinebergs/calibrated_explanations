from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.explanations.reject import RejectPolicy, RejectResult


def _train_wrapper_for_reject_prediction(seed: int = 42):
    X, y = make_classification(n_samples=160, n_features=5, random_state=seed)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.4, random_state=seed)
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=seed))
    wrapper.fit(X_train, y_train)
    wrapper.calibrate(X_cal, y_cal, seed=seed)
    wrapper.explainer.reject_orchestrator.initialize_reject_learner()
    return wrapper, X_cal[:8]


def _make_breakdown(rejected: list[bool]) -> dict[str, object]:
    rejected_mask = np.asarray(rejected, dtype=bool)
    prediction_set = np.array(
        [[True, True] if is_rejected else [True, False] for is_rejected in rejected_mask],
        dtype=bool,
    )
    prediction_set_size = np.sum(prediction_set, axis=1, dtype=int)
    rejected_count = int(np.sum(rejected_mask))
    return {
        "rejected": rejected_mask,
        "error_rate": 0.1,
        "error_rate_defined": True,
        "reject_rate": rejected_count / len(rejected_mask),
        "ambiguity_rate": rejected_count / len(rejected_mask),
        "novelty_rate": 0.0,
        "ambiguity": np.array(rejected_mask, copy=True),
        "novelty": np.zeros(len(rejected_mask), dtype=bool),
        "prediction_set_size": prediction_set_size,
        "prediction_set": prediction_set,
        "epsilon": 0.05,
        "raw_total_examples": int(len(rejected_mask)),
        "raw_reject_counts": {
            "rejected": rejected_count,
            "ambiguity_mask": rejected_count,
            "novelty_mask": 0,
            "prediction_set_size": int(np.sum(prediction_set_size)),
        },
    }


def test_prediction_internals_are_removed_from_calibrated_explainer():
    assert not hasattr(CalibratedExplainer, "predict_internal")
    assert not hasattr(CalibratedExplainer, "predict_calibrated")
    assert not hasattr(CalibratedExplainer, "_predict")


@pytest.mark.parametrize(
    ("policy", "expected_indices"),
    [
        (RejectPolicy.ONLY_ACCEPTED, [0, 2, 4, 6]),
        (RejectPolicy.ONLY_REJECTED, [1, 3, 5, 7]),
    ],
)
def test_prediction_envelope_subset_metadata_matches_full_batch_prediction_payload(
    monkeypatch,
    policy,
    expected_indices,
):
    wrapper, x_query = _train_wrapper_for_reject_prediction()
    baseline_prediction = wrapper.predict(x_query)
    baseline_proba = wrapper.predict_proba(x_query, uq_interval=False)
    breakdown = _make_breakdown([False, True, False, True, False, True, False, True])

    monkeypatch.setattr(
        wrapper.explainer.reject_orchestrator,
        "predict_reject_breakdown",
        lambda *args, **kwargs: breakdown,
    )

    prediction_result = wrapper.predict(x_query, reject_policy=policy)
    probability_result = wrapper.predict_proba(
        x_query,
        uq_interval=False,
        reject_policy=policy,
    )

    assert isinstance(prediction_result, RejectResult)
    assert isinstance(probability_result, RejectResult)

    for result, baseline in (
        (prediction_result, baseline_prediction),
        (probability_result, baseline_proba),
    ):
        assert result.metadata is not None
        assert result.metadata["source_indices"] == expected_indices
        assert result.metadata["matched_count"] == len(expected_indices)
        assert result.metadata["original_count"] == len(x_query)
        assert len(np.asarray(result.prediction)) == len(x_query)
        np.testing.assert_array_equal(
            np.asarray(result.prediction)[expected_indices],
            np.asarray(baseline)[expected_indices],
        )
