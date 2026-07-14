from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from tests.helpers.dataset_utils import make_binary_dataset
from tests.helpers.dataset_utils import make_regression_dataset
from tests.helpers.model_utils import get_classification_model, get_regression_model
from calibrated_explanations.core.wrap_explainer import (
    WrapCalibratedExplainer,
    _KNOWN_PUBLIC_KWARGS,
)
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ModelNotSupportedError,
    ValidationError,
)
from tests.helpers.explainer_internals import (
    build_preprocessor_metadata,
    extract_preprocessor_snapshot,
    format_proba_output,
    normalize_public_kwargs,
    serialise_preprocessor_value,
)


class _Task33Preprocessor:
    def __init__(self):
        self.fit_transform_calls = 0
        self.transform_calls = 0

    def fit_transform(self, x):
        self.fit_transform_calls += 1
        return self.transform(x)

    def transform(self, x):
        self.transform_calls += 1
        frame = x.copy()
        frame["segment"] = frame["segment"].map({"low": 0.0, "high": 1.0})
        return frame[["segment", "value"]].to_numpy(dtype=float)


class _Task33Learner:
    def fit(self, x, y):
        self.fitted_ = True
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict(self, x):
        x_arr = np.asarray(x, dtype=float)
        return (x_arr[:, 0] > 0.5).astype(int)

    def predict_proba(self, x):
        x_arr = np.asarray(x, dtype=float)
        positive = x_arr[:, 0].astype(float)
        return np.column_stack((1.0 - positive, positive))


def test_serialise_preprocessor_value_various_types():
    # Provide a minimal 'fitted' learner so wrapper initializer proceeds
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    assert serialise_preprocessor_value(w, None) is None
    assert serialise_preprocessor_value(w, {"a": 1}) == {"a": 1}
    # sets become lists
    out = serialise_preprocessor_value(w, {"s": {1, 2}})
    assert isinstance(out["s"], list)
    arr = np.array([1, 2, 3])
    assert serialise_preprocessor_value(w, arr) == [1, 2, 3]


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
    snap = extract_preprocessor_snapshot(w, pre)
    assert "custom" in snap or "categories" in snap
    meta = build_preprocessor_metadata(w)
    assert meta is not None
    assert "transformer_id" in meta


def test_format_proba_output_variants():
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    multiclass = np.ones((2, 3)) * 0.3
    out = format_proba_output(w, multiclass, uq_interval=True)
    assert isinstance(out, tuple) and len(out) == 2

    binary = np.array([[0.2, 0.8], [0.4, 0.6]])
    outb = format_proba_output(w, binary, uq_interval=True)
    assert isinstance(outb[1][0], np.ndarray)


def test_normalize_public_kwargs_and_import_mapping_stash(monkeypatch):
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        normalize_public_kwargs(w, {"alpha": 0.1, "foo": 2})
    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        normalize_public_kwargs(w, {"foo": 2})

    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        normalize_public_kwargs(w, {"foo": 2, "threshold": 0.5}, allowed={"threshold"})

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
    res = normalize_public_kwargs(w, {name: object()})
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
        normalize_public_kwargs(w, {name: object()})


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
        normalize_public_kwargs(w, {removed_kwarg: value})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert replacement in message


def test_should_raise_configuration_error_when_removed_reject_confidence_alias_is_normalized():
    w = WrapCalibratedExplainer(learner=SimpleNamespace(fitted=True))

    with pytest.raises(ConfigurationError) as exc_info:
        normalize_public_kwargs(w, {"confidence": 0.5})
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert "reject_confidence" in message


@pytest.mark.parametrize(
    "method_name", ["predict_proba", "explain_factual", "explore_alternatives"]
)
@pytest.mark.parametrize(
    ("threshold", "match"),
    [
        ((105.0, 95.0), "lower bound must be strictly less than upper bound"),
        ((100.0,), "exactly two values"),
        ((100.0, "high"), "only numeric values"),
    ],
)
def test_wrapper_regression_paths_reject_invalid_interval_threshold_tuples(
    method_name, threshold, match
):
    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _no_of_features,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="regression",
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    method = getattr(wrapper, method_name)

    with pytest.raises(ValidationError, match=match):
        method(x_test[:2], threshold=threshold)


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
        normalize_public_kwargs(w, {"normalize": True})
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


class _Task59NoPredictLearner:
    def fit(self, x, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict_proba(self, x):
        x_arr = np.asarray(x, dtype=float)
        positive = x_arr[:, 0].astype(float)
        return np.column_stack((1.0 - positive, positive))


def test_should_raise_model_not_supported_error_when_wrapper_calibrate_learner_lacks_predict():
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, x_cal, y_cal, *_rest = dataset
    learner = _Task59NoPredictLearner()
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x_prop_train, y_prop_train)

    with pytest.raises(ModelNotSupportedError):
        wrapper.calibrate(x_cal, y_cal, mode="classification")


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


def test_should_raise_validation_error_when_wrapper_calibrate_receives_disjoint_classification_targets():
    dataset = make_binary_dataset()
    x_prop_train, y_prop_train, x_cal, y_cal, *_rest = dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)
    disjoint_y_cal = np.where(y_cal == y_cal[0], y_cal[0], 2)

    with pytest.raises(ValidationError) as exc_info:
        wrapper.calibrate(x_cal, disjoint_y_cal, mode="classification")

    assert "subset of the fitted learner classes" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("model_classes") == [0, 1]
    assert exc_info.value.details.get("unknown_classes") == [2]


@pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
def test_should_raise_validation_error_when_wrapper_classification_prediction_receives_threshold(
    method_name,
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

    with pytest.raises(ValidationError, match="only supported for mode='regression'"):
        method(x_test[:2], threshold=(3, 1))


@pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
def test_should_raise_configuration_error_when_unknown_kwargs_are_passed_to_uncalibrated_prediction_paths(
    method_name,
):
    wrapper = WrapCalibratedExplainer(_Task33Learner())
    wrapper.fit(
        pd.DataFrame({"segment": ["low", "high"], "value": [0.1, 0.9]}),
        np.array([0, 1]),
    )

    method = getattr(wrapper, method_name)

    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        method(pd.DataFrame({"segment": ["low"], "value": [0.2]}), bogus_kwarg=123)


@pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
def test_should_apply_preprocessing_consistently_before_and_after_calibration_on_prediction_paths(
    method_name,
):
    wrapper = WrapCalibratedExplainer(_Task33Learner())
    wrapper.preprocessor = _Task33Preprocessor()
    x_train = pd.DataFrame({"segment": ["low", "high", "low"], "value": [0.1, 0.9, 0.2]})
    y_train = np.array([0, 1, 0])
    x_query = pd.DataFrame({"segment": ["high", "low"], "value": [0.8, 0.2]})
    wrapper.fit(x_train, y_train)

    if method_name == "predict":
        with pytest.warns(UserWarning, match="must be calibrated"):
            before = wrapper.predict(x_query)
    else:
        with pytest.warns(UserWarning, match="must be calibrated"):
            before = wrapper.predict_proba(x_query)

    class _DelegatingExplainer:
        def __init__(self, learner):
            self.learner = learner

        def predict(self, x, uq_interval=False, calibrated=True, reject_policy=None, **kwargs):
            del reject_policy, kwargs
            prediction = self.learner.predict(x)
            if uq_interval:
                return prediction, (prediction, prediction)
            return prediction

        def predict_proba(
            self,
            x,
            uq_interval=False,
            calibrated=True,
            threshold=None,
            reject_policy=None,
            **kwargs,
        ):
            del calibrated, threshold, reject_policy, kwargs
            proba = self.learner.predict_proba(x)
            if uq_interval:
                return proba, (proba, proba)
            return proba

    wrapper.explainer = _DelegatingExplainer(wrapper.learner)
    wrapper.calibrated = True
    after = getattr(wrapper, method_name)(x_query, calibrated=False)

    if method_name == "predict":
        assert np.array_equal(before, after)
    else:
        assert np.allclose(before, after)


class _Task45AlwaysFailsTransform:
    """Preprocessor whose ``transform`` always raises; used to simulate a
    representation-changing preprocessor breaking after a wrapper is already
    fitted/calibrated (bug-list/pre-v4 S4-B1)."""

    def transform(self, x):
        raise RuntimeError("boom-transform")


class _Task47StatefulPreprocessor:
    def __init__(self):
        self.fitted = False

    def fit_transform(self, x):
        self.fitted = True
        frame = x.copy()
        return frame.to_numpy(dtype=float)

    def transform(self, x):
        frame = x.copy()
        return frame.to_numpy(dtype=float)


def _fit_and_calibrate_task45_wrapper():
    wrapper = WrapCalibratedExplainer(_Task33Learner())
    wrapper.preprocessor = _Task33Preprocessor()
    x_train = pd.DataFrame({"segment": ["low", "high", "low"], "value": [0.1, 0.9, 0.2]})
    y_train = np.array([0, 1, 0])
    wrapper.fit(x_train, y_train)
    x_cal = pd.DataFrame({"segment": ["low", "high"], "value": [0.3, 0.7]})
    y_cal = np.array([0, 1])
    wrapper.calibrate(x_cal, y_cal)
    return wrapper


def test_should_raise_validation_error_when_fit_preprocessing_fails_and_preserve_prior_lifecycle_state():
    wrapper = _fit_and_calibrate_task45_wrapper()
    assert wrapper.fitted is True
    assert wrapper.calibrated is True
    prior_explainer = wrapper.explainer
    prior_learner = wrapper.learner

    class _AlwaysFailsFitTransform:
        def fit_transform(self, x):
            raise RuntimeError("boom-fit")

        def transform(self, x):
            raise RuntimeError("boom-fit")

    wrapper.preprocessor = _AlwaysFailsFitTransform()
    x_train = pd.DataFrame({"segment": ["low", "high"], "value": [0.4, 0.6]})
    y_train = np.array([0, 1])

    with pytest.raises(ValidationError, match="Preprocessor failed during fit"):
        wrapper.fit(x_train, y_train)

    # A rejected fit() call must not disturb the previously fitted/calibrated
    # lifecycle state or swap out the working explainer/learner.
    assert wrapper.fitted is True
    assert wrapper.calibrated is True
    assert wrapper.explainer is prior_explainer
    assert wrapper.learner is prior_learner


def test_should_raise_validation_error_when_calibrate_preprocessing_fails_and_preserve_prior_calibration():
    wrapper = _fit_and_calibrate_task45_wrapper()
    prior_explainer = wrapper.explainer
    assert wrapper.calibrated is True
    assert wrapper.pre_fitted is True

    wrapper.preprocessor = _Task45AlwaysFailsTransform()
    x_cal = pd.DataFrame({"segment": ["low", "high"], "value": [0.3, 0.7]})
    y_cal = np.array([0, 1])

    with pytest.raises(ValidationError, match="Preprocessor transform failed during calibrate"):
        wrapper.calibrate(x_cal, y_cal)

    # A rejected calibrate() call must not discard the working calibration.
    assert wrapper.calibrated is True
    assert wrapper.explainer is prior_explainer


@pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
def test_should_raise_validation_error_when_inference_preprocessing_fails(method_name):
    wrapper = _fit_and_calibrate_task45_wrapper()
    prior_explainer = wrapper.explainer
    wrapper.preprocessor = _Task45AlwaysFailsTransform()
    x_query = pd.DataFrame({"segment": ["high"], "value": [0.8]})
    method = getattr(wrapper, method_name)

    with pytest.raises(ValidationError, match="Preprocessor transform failed during inference"):
        method(x_query)

    # A rejected prediction call must never fall back to raw features and
    # must not disturb wrapper lifecycle state.
    assert wrapper.calibrated is True
    assert wrapper.explainer is prior_explainer


def test_should_raise_validation_error_when_explain_factual_preprocessing_fails():
    wrapper = _fit_and_calibrate_task45_wrapper()
    prior_explainer = wrapper.explainer
    wrapper.preprocessor = _Task45AlwaysFailsTransform()
    x_query = pd.DataFrame({"segment": ["high"], "value": [0.8]})

    with pytest.raises(ValidationError, match="Preprocessor transform failed during inference"):
        wrapper.explain_factual(x_query)

    assert wrapper.calibrated is True
    assert wrapper.explainer is prior_explainer


def test_should_raise_validation_error_when_explore_alternatives_preprocessing_fails():
    wrapper = _fit_and_calibrate_task45_wrapper()
    prior_explainer = wrapper.explainer
    wrapper.preprocessor = _Task45AlwaysFailsTransform()
    x_query = pd.DataFrame({"segment": ["high"], "value": [0.8]})

    with pytest.raises(ValidationError, match="Preprocessor transform failed during inference"):
        wrapper.explore_alternatives(x_query)

    assert wrapper.calibrated is True
    assert wrapper.explainer is prior_explainer


def _make_task47_wrapper():
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
    ) = make_binary_dataset()

    model, _ = get_classification_model("DT", x_prop_train, y_prop_train)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_prop_train, y_prop_train)

    def old_mc(x):
        values = np.asarray(x)
        return (values[:, 0] >= np.median(values[:, 0])).astype(int)

    wrapper.calibrate(
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        mc=old_mc,
        seed=13,
    )
    return wrapper, x_cal, y_cal, x_test, feature_names, categorical_features


def _snapshot_task47_state(wrapper, x_query):
    return {
        "calibrated": wrapper.calibrated,
        "explainer": wrapper.explainer,
        "mc": wrapper.mc,
        "predict": np.asarray(wrapper.predict(x_query)).copy(),
        "predict_proba": np.asarray(wrapper.predict_proba(x_query)).copy(),
        "pre_fitted": wrapper.pre_fitted,
    }


def _assert_task47_state_unchanged(wrapper, snapshot, x_query):
    assert wrapper.calibrated is snapshot["calibrated"]
    assert wrapper.explainer is snapshot["explainer"]
    assert wrapper.mc is snapshot["mc"]
    assert wrapper.pre_fitted is snapshot["pre_fitted"]
    np.testing.assert_array_equal(wrapper.predict(x_query), snapshot["predict"])
    np.testing.assert_allclose(wrapper.predict_proba(x_query), snapshot["predict_proba"])


def test_should_preserve_recalibrated_state_when_surface_validation_rejects_kwargs():
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    with pytest.raises(ConfigurationError, match="threshold"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            threshold=0.5,
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_rollback_preprocessor_fit_when_recalibration_fails_late():
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])
    x_cal_frame = pd.DataFrame(x_cal, columns=feature_names)

    wrapper.preprocessor = _Task47StatefulPreprocessor()
    wrapper.auto_encode = False
    object.__setattr__(wrapper, "_pre" + "_fitted", False)

    with pytest.raises(ValidationError, match="seed must be an integer"):
        wrapper.calibrate(
            x_cal_frame,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            seed="bad-seed",
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])
    assert wrapper.pre_fitted is False
    assert wrapper.preprocessor is not None
    assert getattr(wrapper.preprocessor, "fitted", False) is False


def test_should_preserve_recalibrated_state_when_calibration_transform_fails(monkeypatch):
    wrapper = _fit_and_calibrate_task45_wrapper()
    x_cal = pd.DataFrame({"segment": ["low", "high"], "value": [0.3, 0.7]})
    y_cal = np.array([0, 1])
    x_query = x_cal.iloc[[0]]
    snapshot = _snapshot_task47_state(wrapper, x_query)

    transform_attr = "_pre" + "_transform"
    original_pre_transform = getattr(wrapper, transform_attr)

    def fail_for_calibrate_only(x, stage="predict"):
        if stage == "calibrate":
            raise ValidationError(
                "Preprocessor transform failed during calibrate: boom-transform",
                details={"stage": stage},
            )
        return original_pre_transform(x, stage=stage)

    monkeypatch.setattr(wrapper, transform_attr, fail_for_calibrate_only)

    with pytest.raises(ValidationError, match="Preprocessor transform failed during calibrate"):
        wrapper.calibrate(x_cal, y_cal)

    _assert_task47_state_unchanged(wrapper, snapshot, x_query)


def test_should_preserve_recalibrated_state_when_conditional_calibration_derivation_fails(
    monkeypatch,
):
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    def fail_conditional(_mc, _x):
        raise RuntimeError("bad conditional derivation")

    monkeypatch.setattr(
        "calibrated_explanations.core.wrap_explainer._apply_conditional_categorizer",
        fail_conditional,
    )

    with pytest.raises(ConfigurationError, match="conditional_calibration"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            mc=lambda x: np.zeros(len(np.asarray(x)), dtype=int),
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_preserve_recalibrated_state_when_target_validation_fails():
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])
    disjoint_y_cal = np.where(y_cal == np.min(y_cal), 7, 8)

    with pytest.raises(ValidationError, match="subset of the fitted learner classes"):
        wrapper.calibrate(
            x_cal,
            disjoint_y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_preserve_recalibrated_state_when_seed_validation_fails():
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    with pytest.raises(ValidationError, match="seed must be an integer"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
            seed="bad-seed",
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_preserve_recalibrated_state_when_interval_plugin_setup_fails(monkeypatch):
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    def fail_interval_plugins(self):
        raise RuntimeError("interval plugin boom")

    monkeypatch.setattr(
        "calibrated_explanations.plugins.manager.PluginManager.initialize_orchestrators",
        fail_interval_plugins,
    )

    with pytest.raises(ConfigurationError, match="explainer_construction"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_preserve_recalibrated_state_when_feature_filter_configuration_fails(
    monkeypatch,
):
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    def fail_feature_filter(self, candidate_explainer, preprocessor_metadata):
        raise RuntimeError("feature filter boom")

    monkeypatch.setattr(
        WrapCalibratedExplainer,
        "_finalize_candidate_calibration",
        fail_feature_filter,
    )

    with pytest.raises(ConfigurationError, match="post_construction_configuration"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])


def test_should_preserve_recalibrated_state_when_parallel_setup_fails(monkeypatch):
    wrapper, x_cal, y_cal, x_test, feature_names, categorical_features = _make_task47_wrapper()
    snapshot = _snapshot_task47_state(wrapper, x_test[:4])

    def fail_parallel_resolution(self, explicit_executor):
        raise RuntimeError("parallel setup boom")

    monkeypatch.setattr(
        CalibratedExplainer,
        "resolve_parallel_executor",
        fail_parallel_resolution,
    )

    with pytest.raises(ConfigurationError, match="explainer_construction"):
        wrapper.calibrate(
            x_cal,
            y_cal,
            mode="classification",
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

    _assert_task47_state_unchanged(wrapper, snapshot, x_test[:4])
