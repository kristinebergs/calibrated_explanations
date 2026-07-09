from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from tests.helpers.dataset_utils import make_binary_dataset
from tests.helpers.model_utils import get_classification_model
from calibrated_explanations.core.wrap_explainer import (
    WrapCalibratedExplainer,
    _KNOWN_PUBLIC_KWARGS,
)
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)


def test_serialise_preprocessor_value_various_types():
    # Provide a minimal 'fitted' learner so wrapper initializer proceeds
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    assert w.serialise_preprocessor_value(None) is None
    assert w.serialise_preprocessor_value({"a": 1}) == {"a": 1}
    # sets become lists
    out = w.serialise_preprocessor_value({"s": {1, 2}})
    assert isinstance(out["s"], list)
    arr = np.array([1, 2, 3])
    assert w.serialise_preprocessor_value(arr) == [1, 2, 3]


def test_extract_preprocessor_snapshot_and_build_metadata():
    class DummyTransformer:
        pass

    class Pre:
        def get_mapping_snapshot(self):
            return {"m": 1}

        categories_ = [["a"]]

        transformers_ = [("t", DummyTransformer(), [0])]

        def get_feature_names_out(self):
            return ["f0"]

        mapping_ = {"x": 1}

    pre = Pre()
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    w.auto_encode = True
    w.preprocessor = pre
    snap = w.extract_preprocessor_snapshot(pre)
    assert "custom" in snap or "categories" in snap
    meta = w.build_preprocessor_metadata()
    assert meta is not None
    assert "transformer_id" in meta


def test_format_proba_output_variants():
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    multiclass = np.ones((2, 3)) * 0.3
    out = w.format_proba_output(multiclass, uq_interval=True)
    assert isinstance(out, tuple) and len(out) == 2

    binary = np.array([[0.2, 0.8], [0.4, 0.6]])
    outb = w.format_proba_output(binary, uq_interval=True)
    assert isinstance(outb[1][0], np.ndarray)


def test_normalize_public_kwargs_and_import_mapping_stash(monkeypatch):
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        w.normalize_public_kwargs({"alpha": 0.1, "foo": 2})
    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        w.normalize_public_kwargs({"foo": 2})

    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        w.normalize_public_kwargs({"foo": 2, "threshold": 0.5}, allowed={"threshold"})

    # import_preprocessor_mapping should warn when mapping cannot be applied
    mapping = {"a": 1}
    with pytest.warns(UserWarning):
        w.import_preprocessor_mapping(mapping)
    # We do not introspect private stash attributes; only ensure a warning was raised.


@pytest.mark.parametrize(
    "name",
    [
        "categorical_labels",
        "class_labels",
        "features_to_ignore",
        "factual_plugin",
        "alternative_plugin",
        "fast_plugin",
        "interval_plugin",
        "fast_interval_plugin",
        "plot_style",
        "oob",
    ],
)
def test_normalize_public_kwargs_accepts_documented_names(name):
    """ADR-038 D3 audit: these names are documented/used in notebooks and docs
    but were previously absent from ``_KNOWN_PUBLIC_KWARGS``, so the pre-Task-5
    warn-and-forward behavior masked that they would have failed once the
    unknown-kwarg policy became fail-fast."""
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    assert name in _KNOWN_PUBLIC_KWARGS
    res = w.normalize_public_kwargs({name: object()})
    assert name in res


@pytest.mark.parametrize(
    "name",
    [
        "condition",
        "condition_label",
        "condition_labels",
        "include_reject_details",
        "output_interval",
        "y_threshold",
    ],
)
def test_normalize_public_kwargs_rejects_dead_and_internal_only_names(name):
    """ADR-038 5A: these names were removed from ``_KNOWN_PUBLIC_KWARGS`` because
    they either have no consumer anywhere in src/ (dead names left over from the
    v0.11.4 Task 15 enumeration) or are set internally by
    ``CalibratedExplainer.predict_proba`` for its own interval-learner calls and
    conflict if a caller passes them directly. They must now be rejected like any
    other unrecognized name, not silently accepted."""
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    assert name not in _KNOWN_PUBLIC_KWARGS
    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        w.normalize_public_kwargs({name: object()})


@pytest.mark.parametrize(
    ("removed_kwarg", "value", "replacement"),
    [
        ("guarded", True, "guarded_options=GuardedOptions()"),
        ("significance", 0.1, "GuardedOptions(confidence=1-significance)"),
        ("n_neighbors", 5, "GuardedOptions(n_neighbors=...)"),
        ("normalize_guard", True, "GuardedOptions(normalize=...)"),
        ("merge_adjacent", True, "GuardedOptions(merge_adjacent=...)"),
    ],
)
def test_should_raise_configuration_error_when_removed_guarded_kwarg_is_normalized(
    removed_kwarg, value, replacement
):
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))

    with pytest.raises(ConfigurationError) as exc_info:
        w.normalize_public_kwargs({removed_kwarg: value})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert replacement in message


def test_should_raise_configuration_error_when_removed_reject_confidence_alias_is_normalized():
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))

    with pytest.raises(ConfigurationError) as exc_info:
        w.normalize_public_kwargs({"confidence": 0.5})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert "reject_confidence" in message


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives", "explain_fast"])
@pytest.mark.parametrize(
    ("removed_kwarg", "value", "replacement"),
    [
        ("guarded", True, "guarded_options=GuardedOptions()"),
        ("significance", 0.1, "GuardedOptions(confidence=1-significance)"),
        ("n_neighbors", 5, "GuardedOptions(n_neighbors=...)"),
        ("normalize_guard", True, "GuardedOptions(normalize=...)"),
        ("merge_adjacent", True, "GuardedOptions(merge_adjacent=...)"),
    ],
)
def test_should_raise_configuration_error_when_removed_guarded_kwargs_are_passed_through_wrapper_delegation(
    method_name, removed_kwarg, value, replacement
):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    method = getattr(wrapper, method_name)

    with pytest.raises(ConfigurationError) as exc_info:
        method(x_test[:2], **{removed_kwarg: value})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert "GuardedOptions" in message
    assert replacement in message


def test_known_public_kwargs_should_not_reintroduce_removed_guarded_names():
    removed_names = {"confidence", "guarded", "n_neighbors", "merge_adjacent"}
    assert removed_names.isdisjoint(_KNOWN_PUBLIC_KWARGS)


def test_should_raise_configuration_error_when_removed_normalize_alias_is_normalized():
    """ADR-038 5B: normalize= looked like a synonym of normalization= but was a
    removed-alias trap; it must fail fast at the wrapper gate, not deep inside
    VennAbers.predict_proba."""
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))

    with pytest.raises(ConfigurationError) as exc_info:
        w.normalize_public_kwargs({"normalize": True})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert "normalization" in message
    assert "normalize" not in _KNOWN_PUBLIC_KWARGS


def test_mondrian_categorizer_alias_resolves_to_mc():
    """ADR-038 5B: mondrian_categorizer is an intentional, documented alias for mc."""
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        _x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)

    categorizer = lambda row: 0  # noqa: E731 - trivial single-bin categorizer for the test
    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        mondrian_categorizer=categorizer,
    )
    assert wrapper.mc is categorizer


def test_mc_and_mondrian_categorizer_conflict_raises():
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        _x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)

    with pytest.raises(ConfigurationError, match="not both"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            mc=lambda row: 0,
            mondrian_categorizer=lambda row: 0,
        )


@pytest.mark.parametrize(
    ("method_name", "kwarg_name", "kwarg_value"),
    [
        ("calibrate", "guarded_options", object()),
        ("calibrate", "reject_confidence", 0.9),
        ("calibrate", "threshold", 0.5),
        ("calibrate", "classes", [0, 1]),
        ("predict", "mode", "regression"),
        ("predict", "guarded_options", object()),
        ("predict", "categorical_features", [0]),
        ("predict_proba", "low_high_percentiles", (5, 95)),
        ("predict_proba", "classes", [0, 1]),
        ("explain_factual", "mode", "regression"),
        ("explain_factual", "oob", True),
        ("explain_fast", "guarded_options", object()),
        ("explain_fast", "multi_labels_enabled", True),
    ],
)
def test_should_raise_configuration_error_for_cross_method_kwargs(
    method_name, kwarg_name, kwarg_value
):
    """ADR-038 5B: a name recognized on one wrapper method but not applicable to
    another must now raise ConfigurationError instead of being silently accepted
    and ignored (the root cause behind the normalize/normalization confusion)."""
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    if method_name == "calibrate":
        with pytest.raises(ConfigurationError, match="not valid here"):
            wrapper.calibrate(
                x_cal,
                y_cal,
                mode="classification",
                feature_names=feature_names,
                categorical_features=categorical_features,
                **{kwarg_name: kwarg_value},
            )
    else:
        method = getattr(wrapper, method_name)
        with pytest.raises(ConfigurationError, match="not valid here"):
            method(x_test[:2], **{kwarg_name: kwarg_value})


@pytest.mark.parametrize(
    ("x_empty", "y_empty"),
    [
        (np.empty((0, 3)), np.empty((0,))),
        (pd.DataFrame(columns=["f0", "f1", "f2"]), pd.Series(dtype=float)),
    ],
)
def test_should_raise_data_shape_error_when_wrapper_calibrate_receives_empty_calibration_data(
    x_empty, y_empty
):
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, *_rest = dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)

    with pytest.raises(DataShapeError) as exc_info:
        wrapper.calibrate(x_empty, y_empty, mode="classification")

    assert "at least one sample" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("requirement") == "non-empty calibration data"


def test_should_raise_data_shape_error_when_core_explainer_receives_empty_calibration_data():
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, *_rest = dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)

    with pytest.raises(DataShapeError) as exc_info:
        CalibratedExplainer(
            model, np.empty((0, x_prop_train.shape[1])), np.empty((0,)), mode="classification"
        )

    assert "at least one sample" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("requirement") == "non-empty calibration data"


def test_should_raise_validation_error_when_wrapper_calibrate_receives_single_class_targets():
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, x_cal, y_cal, *_rest = dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)
    single_class_mask = y_cal == y_cal[0]

    with pytest.raises(ValidationError) as exc_info:
        wrapper.calibrate(x_cal[single_class_mask], y_cal[single_class_mask], mode="classification")

    assert "at least two unique target classes" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("unique_class_count") == 1
    assert exc_info.value.details.get("model_class_count") == 2


def test_should_raise_validation_error_when_core_explainer_receives_single_class_targets():
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, x_cal, y_cal, *_rest = dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    single_class_mask = y_cal == y_cal[0]

    with pytest.raises(ValidationError) as exc_info:
        CalibratedExplainer(
            model,
            x_cal[single_class_mask],
            y_cal[single_class_mask],
            mode="classification",
        )

    assert "at least two unique target classes" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("unique_class_count") == 1
    assert exc_info.value.details.get("model_class_count") == 2
