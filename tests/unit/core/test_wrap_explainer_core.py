import numpy as np
from types import SimpleNamespace

import pytest

from tests.helpers.dataset_utils import make_binary_dataset
from tests.helpers.model_utils import get_classification_model
from calibrated_explanations.core.wrap_explainer import (
    WrapCalibratedExplainer,
    _KNOWN_PUBLIC_KWARGS,
)
from calibrated_explanations.utils.exceptions import ConfigurationError


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
    with pytest.warns(UserWarning, match="unknown keyword arguments"):
        res = w.normalize_public_kwargs({"foo": 2})
    assert res.get("foo") == 2

    with pytest.warns(UserWarning, match="unknown keyword arguments"):
        filtered = w.normalize_public_kwargs({"foo": 2, "threshold": 0.5}, allowed={"threshold"})
    assert filtered == {"threshold": 0.5}

    # import_preprocessor_mapping should warn when mapping cannot be applied
    mapping = {"a": 1}
    with pytest.warns(UserWarning):
        w.import_preprocessor_mapping(mapping)
    # We do not introspect private stash attributes; only ensure a warning was raised.


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
