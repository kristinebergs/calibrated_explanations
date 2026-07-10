from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.discretizer_config import (
    discretizer_is_cached,
    instantiate_discretizer,
    setup_discretized_data,
    validate_discretizer_choice,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_validate_discretizer_choice__should_raise_when_invalid_for_regression():
    with pytest.raises(ValidationError, match="discretizer must be"):
        validate_discretizer_choice("entropy", mode="regression")


def test_validate_discretizer_choice__should_raise_when_invalid_for_classification():
    with pytest.raises(ValidationError, match="discretizer must be"):
        validate_discretizer_choice("regressor", mode="classification")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("classification", "binaryEntropy"),
        ("regression", "binaryRegressor"),
    ],
)
def test_validate_discretizer_choice__should_default_when_none(mode: str, expected: str) -> None:
    assert validate_discretizer_choice(None, mode=mode) == expected


def test_instantiate_discretizer__should_raise_when_condition_source_invalid():
    x_cal = np.zeros((2, 2))
    features_to_ignore = np.asarray([], dtype=int)

    with pytest.raises(ValidationError, match="condition_source must be"):
        instantiate_discretizer(
            "binaryEntropy",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=None,
            y_cal=None,
            seed=0,
            current_discretizer=None,
            condition_source="bad",
        )


def test_instantiate_discretizer__should_raise_when_unknown_discretizer_name():
    x_cal = np.zeros((2, 2))
    features_to_ignore = np.asarray([], dtype=int)

    with pytest.raises(ValidationError, match="Unknown discretizer"):
        instantiate_discretizer(
            "does-not-exist",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=None,
            y_cal=None,
            seed=0,
            current_discretizer=None,
            condition_source="observed",
        )


def test_instantiate_discretizer__should_warn_and_fall_back_to_y_cal_when_prediction_labels_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_cal = np.asarray([[0.0], [1.0]])
    y_cal = np.asarray([0, 1])
    features_to_ignore = np.asarray([], dtype=int)

    class DummyDiscretizer:
        def __init__(self, *_args, **kwargs):
            self.labels = kwargs["labels"]

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.EntropyDiscretizer",
        DummyDiscretizer,
    )

    with pytest.warns(UserWarning, match="falling back to observed y_cal"):
        out = instantiate_discretizer(
            "entropy",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=["x0"],
            y_cal=y_cal,
            seed=0,
            current_discretizer=None,
            condition_source="prediction",
            condition_labels=None,
        )

    assert out.labels.tolist() == [0, 1]


def test_instantiate_discretizer__should_warn_and_fall_back_to_y_cal_when_condition_labels_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_cal = np.asarray([[0.0], [1.0]])
    y_cal = np.asarray([2, 3])
    features_to_ignore = np.asarray([], dtype=int)

    class DummyDiscretizer:
        def __init__(self, *_args, **kwargs):
            self.labels = kwargs["labels"]

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.EntropyDiscretizer",
        DummyDiscretizer,
    )

    with pytest.warns(UserWarning, match="No usable condition labels available"):
        out = instantiate_discretizer(
            "entropy",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=["x0"],
            y_cal=y_cal,
            seed=0,
            current_discretizer=None,
            condition_source="prediction",
            condition_labels=np.asarray([]),
        )

    assert out.labels.tolist() == [2, 3]


@pytest.mark.parametrize(
    ("discretizer_name", "symbol_name"),
    [
        ("binaryEntropy", "BinaryEntropyDiscretizer"),
        ("entropy", "EntropyDiscretizer"),
        ("binaryRegressor", "BinaryRegressorDiscretizer"),
        ("regressor", "RegressorDiscretizer"),
    ],
)
def test_discretizer_is_cached__should_reflect_matching_instance_type(
    monkeypatch: pytest.MonkeyPatch,
    discretizer_name: str,
    symbol_name: str,
) -> None:
    module = __import__("calibrated_explanations.core.discretizer_config", fromlist=["unused"])
    sentinel_type = type(f"{symbol_name}Sentinel", (), {})

    monkeypatch.setattr(module, symbol_name, sentinel_type)

    assert discretizer_is_cached(discretizer_name, sentinel_type()) is True
    assert discretizer_is_cached(discretizer_name, object()) is False
    assert discretizer_is_cached(discretizer_name, None) is False


@pytest.mark.parametrize(
    ("discretizer_name", "symbol_name"),
    [
        ("binaryEntropy", "BinaryEntropyDiscretizer"),
        ("binaryRegressor", "BinaryRegressorDiscretizer"),
        ("entropy", "EntropyDiscretizer"),
        ("regressor", "RegressorDiscretizer"),
    ],
)
def test_instantiate_discretizer__should_return_cached_instance_when_type_matches(
    monkeypatch: pytest.MonkeyPatch,
    discretizer_name: str,
    symbol_name: str,
) -> None:
    module = __import__("calibrated_explanations.core.discretizer_config", fromlist=["unused"])
    sentinel_type = type(f"{symbol_name}Sentinel", (), {})
    sentinel_instance = sentinel_type()

    monkeypatch.setattr(module, symbol_name, sentinel_type)

    out = instantiate_discretizer(
        discretizer_name,
        x_cal=np.asarray([[0.0], [1.0]]),
        features_to_ignore=np.asarray([], dtype=int),
        feature_names=["x0"],
        y_cal=np.asarray([0, 1]),
        seed=0,
        current_discretizer=sentinel_instance,
        condition_source="observed",
    )

    assert out is sentinel_instance


@pytest.mark.parametrize(
    ("discretizer_name", "symbol_name"),
    [
        ("binaryEntropy", "BinaryEntropyDiscretizer"),
        ("binaryRegressor", "BinaryRegressorDiscretizer"),
        ("regressor", "RegressorDiscretizer"),
    ],
)
def test_instantiate_discretizer__should_construct_requested_discretizer_when_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    discretizer_name: str,
    symbol_name: str,
) -> None:
    module = __import__("calibrated_explanations.core.discretizer_config", fromlist=["unused"])

    class DummyDiscretizer:
        def __init__(self, *_args, **kwargs):
            self.labels = kwargs["labels"]
            self.random_state = kwargs["random_state"]

    monkeypatch.setattr(module, symbol_name, DummyDiscretizer)

    out = instantiate_discretizer(
        discretizer_name,
        x_cal=np.asarray([[0.0], [1.0]]),
        features_to_ignore=np.asarray([], dtype=int),
        feature_names=["x0"],
        y_cal=np.asarray([4, 5]),
        seed=7,
        current_discretizer=object(),
        condition_source="observed",
    )

    assert isinstance(out, DummyDiscretizer)
    assert out.labels.tolist() == [4, 5]
    assert out.random_state == 7


def test_setup_discretized_data__should_build_value_frequency_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discretized = np.asarray([[1, 5], [2, 5], [1, 7]], dtype=float)

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config._discretize_func",
        lambda explainer, x: discretized,
    )

    feature_data, returned = setup_discretized_data(
        explainer_instance=object(),
        discretizer=object(),
        x_cal=np.asarray([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        num_features=2,
    )

    assert returned is discretized
    assert feature_data[0]["values"] == [1.0, 2.0]
    assert np.allclose(feature_data[0]["frequencies"], np.asarray([2 / 3, 1 / 3]))
    assert feature_data[1]["values"] == [5.0, 7.0]
    assert np.allclose(feature_data[1]["frequencies"], np.asarray([2 / 3, 1 / 3]))
