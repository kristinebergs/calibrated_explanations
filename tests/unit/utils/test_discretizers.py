from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)


def test_binary_entropy_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)
    with pytest.raises(ValidationError):
        BinaryEntropyDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_binary_entropy_discretizer_falls_back_to_median_split():
    data = np.ones((4, 1)) * 0.5  # Constant feature triggers median fallback.
    labels = np.array([0, 1, 0, 1])
    feature_names = ("feat",)

    discretizer = BinaryEntropyDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    transformed = discretizer.discretize(data.copy())
    assert transformed.shape == data.shape
    assert np.all(transformed == 0)  # Median fallback collapses to a single bin.
    assert discretizer.mins[0][0] == pytest.approx(0.5)
    assert discretizer.maxs[0][-1] == pytest.approx(0.5)


def test_binary_regressor_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)

    with pytest.raises(ValidationError):
        BinaryRegressorDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_binary_regressor_discretizer_handles_constant_feature():
    data = np.full((4, 1), 0.25)
    labels = np.array([0.1, 0.2, 0.3, 0.4])
    feature_names = ("feat",)

    discretizer = BinaryRegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    transformed = discretizer.discretize(data.copy())
    assert transformed.shape == data.shape
    assert np.all(transformed == 0)
    assert discretizer.mins[0][0] == pytest.approx(0.25)
    assert discretizer.maxs[0][-1] == pytest.approx(0.25)


def test_entropy_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)

    with pytest.raises(ValidationError):
        EntropyDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_regressor_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)

    with pytest.raises(ValidationError):
        RegressorDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_binary_discretizer_reprs_match_class_names():
    data = np.array([[0.1], [0.2], [0.3], [0.4]])
    class_labels = np.array([0, 1, 0, 1])
    regression_labels = np.array([0.1, 0.2, 0.3, 0.4])
    feature_names = ("feat",)

    binary_entropy = BinaryEntropyDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=class_labels,
        random_state=0,
    )
    binary_regressor = BinaryRegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=regression_labels,
        random_state=0,
    )

    assert repr(binary_entropy) == "BinaryEntropyDiscretizer()"
    assert repr(binary_regressor) == "BinaryRegressorDiscretizer()"
