from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ValidationError


def _make_label_encoding_fixture(label_space: tuple[object, ...]) -> tuple[np.ndarray, ...]:
    n_classes = len(label_space)
    x, y_numeric = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    y = np.asarray([label_space[int(index)] for index in y_numeric])
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.5, random_state=42, stratify=y
    )
    x_cal, x_test, y_cal, _y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=24, stratify=y_tmp
    )
    feature_names = [f"f{i}" for i in range(x.shape[1])]
    expected_labels = [str(label) for label in np.unique(y)]
    return x_train, x_cal, x_test, y_train, y_cal, np.asarray(expected_labels), feature_names


def _fit_core_explainer(label_space: tuple[object, ...]) -> tuple[CalibratedExplainer, np.ndarray]:
    x_train, x_cal, x_test, y_train, y_cal, _expected_labels, feature_names = (
        _make_label_encoding_fixture(label_space)
    )
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(x_train, y_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
    )
    return explainer, x_test


def _fit_wrapper(label_space: tuple[object, ...]) -> tuple[WrapCalibratedExplainer, np.ndarray]:
    x_train, x_cal, x_test, y_train, y_cal, _expected_labels, feature_names = (
        _make_label_encoding_fixture(label_space)
    )
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_train, y_train)
    wrapper.calibrate(x_cal, y_cal, mode="classification", feature_names=feature_names)
    return wrapper, x_test


@pytest.mark.parametrize(
    ("label_space", "expects_internal_encoding"),
    [
        ((0, 1), False),
        ((1, 2), True),
        ((2, 5, 9), True),
        (("bird", "cat", "dog"), True),
    ],
)
def test_should_keep_core_classification_labels_and_predictions_coherent_for_supported_label_spaces(
    label_space: tuple[object, ...], expects_internal_encoding: bool
) -> None:
    explainer, x_test = _fit_core_explainer(label_space)
    expected_labels = [str(label) for label in np.unique(np.asarray(label_space))]

    assert np.array_equal(np.unique(explainer.y_cal), np.arange(len(label_space)))
    assert set(explainer.class_labels.keys()) == set(range(len(label_space)))
    assert list(explainer.class_labels.values()) == expected_labels
    if expects_internal_encoding:
        assert explainer.label_map is not None
    else:
        assert explainer.label_map is None

    probabilities, (low, high) = explainer.predict_proba(x_test[:6], uq_interval=True)
    explanations = explainer.explain_factual(x_test[:2])

    assert probabilities.shape == (6, len(label_space))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert low.shape[0] == 6
    assert high.shape[0] == 6
    assert len(explanations) == 2


@pytest.mark.parametrize(
    ("label_space", "expects_internal_encoding"),
    [
        ((0, 1), False),
        ((1, 2), True),
        ((2, 5, 9), True),
        (("bird", "cat", "dog"), True),
    ],
)
def test_should_keep_wrapper_classification_labels_and_predictions_coherent_for_supported_label_spaces(
    label_space: tuple[object, ...], expects_internal_encoding: bool
) -> None:
    wrapper, x_test = _fit_wrapper(label_space)
    assert wrapper.explainer is not None
    expected_labels = [str(label) for label in np.unique(np.asarray(label_space))]

    assert np.array_equal(np.unique(wrapper.explainer.y_cal), np.arange(len(label_space)))
    assert set(wrapper.explainer.class_labels.keys()) == set(range(len(label_space)))
    assert list(wrapper.explainer.class_labels.values()) == expected_labels
    if expects_internal_encoding:
        assert wrapper.explainer.label_map is not None
    else:
        assert wrapper.explainer.label_map is None

    probabilities, (low, high) = wrapper.predict_proba(x_test[:6], uq_interval=True)
    explanations = wrapper.explain_factual(x_test[:2])

    assert probabilities.shape == (6, len(label_space))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert low.shape[0] == 6
    assert high.shape[0] == 6
    assert len(explanations) == 2


def test_should_raise_validation_error_when_wrapper_calibration_targets_have_single_class() -> None:
    x_train, x_cal, _x_test, y_train, y_cal, _expected_labels, feature_names = (
        _make_label_encoding_fixture((2, 5, 9))
    )
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_train, y_train)
    single_class = np.asarray([y_cal[0]] * len(y_cal))

    with pytest.raises(ValidationError, match="at least two unique target classes"):
        wrapper.calibrate(x_cal, single_class, mode="classification", feature_names=feature_names)
