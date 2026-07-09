import logging
import numpy as np
import pytest

from tests.helpers.model_utils import get_classification_model, get_regression_model
from tests.helpers.dataset_utils import make_binary_dataset, make_regression_dataset
from tests.helpers.explainer_utils import initiate_explainer
from calibrated_explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.reject import RejectResult, RejectPolicy
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


def test_predict_skip_reject_internal_returns_prediction():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    # When _ce_skip_reject is True the legacy calibrated prediction path is used
    res = cal_exp.predict(x_test, _ce_skip_reject=True)
    # Expect numpy array (formatted classification labels)
    assert isinstance(res, (list, np.ndarray))


def test_predict_with_implicit_default_reject_policy_logs(monkeypatch):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    # Set a non-NONE default to trigger implicit_default_used when no reject_policy provided
    cal_exp.default_reject_policy = RejectPolicy.FLAG

    records = []

    def fake_info(msg, *a, **k):
        records.append(msg)

    monkeypatch.setattr(
        logging.getLogger("calibrated_explanations.core.calibrated_explainer"), "info", fake_info
    )

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test)
    assert isinstance(res, RejectResult)
    assert any("Default reject policy" in str(r) or "Default reject policy" in r for r in records)


def test_predict_rr_prediction_none_preserved(monkeypatch):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
    assert res.prediction is None


def test_predict_proba_uncalibrated_regression_raises_when_threshold():
    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        no_of_features,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    # uncalibrated regression with threshold should raise ValidationError inside helper
    with pytest.raises(Exception):
        cal_exp.predict(x_test, calibrated=False, threshold=0.5)


def test_invalid_default_policy_falls_back_to_legacy_payloads():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )
    cal_exp.default_reject_policy = "not-a-policy"

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        pred = cal_exp.predict(x_test[:4])
    assert not isinstance(pred, RejectResult)

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        proba = cal_exp.predict_proba(x_test[:4], uq_interval=False)
    assert not isinstance(proba, RejectResult)

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        expl = cal_exp.explain_factual(x_test[:2])
    assert not isinstance(expl, RejectResult)
    assert hasattr(expl, "explanations")


def test_invalid_explicit_policy_fails_fast_across_predict_and_explain():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.predict(x_test[:4], reject_policy="not-a-policy")

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.predict_proba(x_test[:4], reject_policy="not-a-policy")

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.explain_factual(x_test[:2], reject_policy="not-a-policy")


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
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
def test_should_fail_fast_when_removed_guarded_kwargs_are_passed_to_core_explain_apis(
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    method = getattr(cal_exp, method_name)
    kwargs = {removed_kwarg: value}

    with pytest.raises(ConfigurationError) as exc_info:
        method(x_test[:2], **kwargs)
    message = str(exc_info.value)
    assert "removed in v0.11.5" in message
    assert "GuardedOptions" in message
    assert replacement in message


def test_reject_confidence_forwarded_across_explain_and_guarded_paths(monkeypatch):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    from calibrated_explanations import GuardedOptions

    seen_confidences = []

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, reject_confidence=0.95, **kwargs):
        seen_confidences.append(float(reject_confidence))
        return RejectResult(
            prediction=None,
            explanation=None,
            rejected=np.zeros(len(x), dtype=bool),
            policy=RejectPolicy.FLAG,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    cal_exp.explain_factual(x_test[:2], reject_policy=RejectPolicy.FLAG, reject_confidence=0.81)
    cal_exp.explore_alternatives(
        x_test[:2], reject_policy=RejectPolicy.FLAG, reject_confidence=0.82
    )
    cal_exp.explain_factual(
        x_test[:2],
        guarded_options=GuardedOptions(),
        reject_policy=RejectPolicy.FLAG,
        reject_confidence=0.83,
    )
    cal_exp.explore_alternatives(
        x_test[:2],
        guarded_options=GuardedOptions(),
        reject_policy=RejectPolicy.FLAG,
        reject_confidence=0.84,
    )

    assert seen_confidences == [0.81, 0.82, 0.83, 0.84]


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_confidence_rejected_across_predict_and_explain(bad_confidence):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.predict(
            x_test[:3], reject_policy=RejectPolicy.FLAG, reject_confidence=bad_confidence
        )
    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.predict_proba(
            x_test[:3], reject_policy=RejectPolicy.FLAG, reject_confidence=bad_confidence
        )
    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.explain_factual(
            x_test[:2], reject_policy=RejectPolicy.FLAG, reject_confidence=bad_confidence
        )


def test_removed_reject_confidence_alias_rejected_across_predict_and_predict_proba():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    with pytest.raises(ConfigurationError, match="reject_confidence"):
        cal_exp.predict(x_test[:3], reject_policy=RejectPolicy.FLAG, confidence=0.5)
    with pytest.raises(ConfigurationError, match="reject_confidence"):
        cal_exp.predict_proba(x_test[:3], reject_policy=RejectPolicy.FLAG, confidence=0.5)


def test_reject_context_uses_source_indices_for_only_accepted(monkeypatch):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    class DummyExplanation:
        def __init__(self, name):
            self.name = name
            self.reject_context = None

    payload = [DummyExplanation("accepted-1"), DummyExplanation("accepted-3")]

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, reject_confidence=0.95, **kwargs):
        return RejectResult(
            prediction=None,
            explanation=payload,
            rejected=np.array([True, False, True, False]),
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={
                "source_indices": [1, 3],
                "original_count": 4,
                "prediction_set_size": np.array([2, 1, 2, 1]),
                "ambiguity_mask": np.array([True, False, True, False]),
                "novelty_mask": np.array([False, False, False, False]),
                "epsilon": 0.05,
            },
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    result = cal_exp.explain_factual(x_test[:4], reject_policy=RejectPolicy.ONLY_ACCEPTED)
    assert isinstance(result, RejectResult)
    assert result.explanation[0].reject_context.rejected is False
    assert result.explanation[1].reject_context.rejected is False


def test_reject_context_fallback_mapping_warns_when_source_indices_missing(monkeypatch):
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    class DummyExplanation:
        def __init__(self):
            self.reject_context = None

    payload = [DummyExplanation(), DummyExplanation()]

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, reject_confidence=0.95, **kwargs):
        return RejectResult(
            prediction=None,
            explanation=payload,
            rejected=np.array([True, False, True, False]),
            policy=RejectPolicy.ONLY_REJECTED,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    with pytest.warns(UserWarning, match="missing source_indices"):
        result = cal_exp.explain_factual(x_test[:4], reject_policy=RejectPolicy.ONLY_REJECTED)
    assert isinstance(result, RejectResult)
    assert result.explanation[0].reject_context.rejected is True
    assert result.explanation[1].reject_context.rejected is True


def test_regression_predict_proba_forwards_threshold_to_reject_policy(monkeypatch):
    dataset = make_regression_dataset()
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

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    seen_thresholds = []

    def fake_apply_policy(
        policy, x, explain_fn=None, bins=None, reject_confidence=0.95, threshold=None, **kwargs
    ):
        seen_thresholds.append(threshold)
        return RejectResult(
            prediction=None,
            explanation=None,
            rejected=np.zeros(len(x), dtype=bool),
            policy=RejectPolicy.FLAG,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)
    cal_exp.predict_proba(x_test[:3], reject_policy=RejectPolicy.FLAG, threshold=0.42)
    assert seen_thresholds == [0.42]


def test_regression_reject_without_threshold_raises_across_paths():
    dataset = make_regression_dataset()
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

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.predict(x_test[:3], reject_policy=RejectPolicy.FLAG)
    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.predict_proba(x_test[:3], reject_policy=RejectPolicy.FLAG)
    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.explain_factual(x_test[:2], reject_policy=RejectPolicy.FLAG)


def test_reject_metadata_contract_present_across_predict_proba_and_explain():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    required = {
        "policy",
        "reject_rate",
        "accepted_count",
        "rejected_count",
        "effective_confidence",
        "effective_threshold",
        "source_indices",
        "original_count",
        "init_ok",
        "fallback_used",
        "init_error",
        "degraded_mode",
    }

    pred = cal_exp.predict(x_test[:6], reject_policy=RejectPolicy.FLAG)
    proba = cal_exp.predict_proba(x_test[:6], reject_policy=RejectPolicy.FLAG, uq_interval=False)
    expl = cal_exp.explain_factual(x_test[:6], reject_policy=RejectPolicy.FLAG)

    assert isinstance(pred, RejectResult)
    assert isinstance(proba, RejectResult)
    assert required.issubset((pred.metadata or {}).keys())
    assert required.issubset((proba.metadata or {}).keys())
    assert required.issubset(expl.metadata.keys())


# --- ADR-038 5C: fail-fast kwarg validation extended to CalibratedExplainer directly ---


def _classification_explainer():
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
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )
    return cal_exp, x_test


def test_should_raise_configuration_error_when_init_receives_unknown_kwarg():
    dataset = make_binary_dataset()
    (x_prop_train, y_prop_train, x_cal, y_cal, _, _, _, _, categorical_features, feature_names) = (
        dataset
    )
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        initiate_explainer(
            model,
            x_cal,
            y_cal,
            feature_names,
            categorical_features,
            mode="classification",
            totally_bogus_kwarg=5,
        )


@pytest.mark.parametrize(
    ("method_name", "kwarg_name", "kwarg_value"),
    [
        ("predict", "guarded_options", object()),
        ("predict", "mode", "regression"),
        ("predict", "feature_names", ["a"]),
        ("predict_proba", "low_high_percentiles", (5, 95)),
        ("predict_proba", "classes", [0, 1]),
        ("predict_proba", "feature", 0),
    ],
)
def test_should_raise_configuration_error_for_cross_method_kwargs_on_closed_surfaces(
    method_name, kwarg_name, kwarg_value
):
    """ADR-038 5C: predict/predict_proba are closed surfaces (no experimental
    exception); a name known elsewhere but not here must raise."""
    cal_exp, x_test = _classification_explainer()
    method = getattr(cal_exp, method_name)
    with pytest.raises(ConfigurationError, match="unknown keyword arguments"):
        method(x_test[:2], **{kwarg_name: kwarg_value})


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
@pytest.mark.parametrize(
    ("kwarg_name", "kwarg_value"),
    [
        ("mode", "regression"),
        ("feature_names", ["a"]),
        ("categorical_features", [0]),
        ("oob", True),
        ("seed", 1),
    ],
)
def test_should_raise_configuration_error_for_cross_surface_kwargs_on_explain_methods(
    method_name, kwarg_name, kwarg_value
):
    """ADR-038 5C: explain_factual/explore_alternatives keep the ADR-038 §3
    experimental plugin-forwarding exception, but a name known on a *closed*
    surface (__init__/predict/predict_proba) must still be rejected here."""
    cal_exp, x_test = _classification_explainer()
    method = getattr(cal_exp, method_name)
    with pytest.raises(ConfigurationError, match="not valid here"):
        method(x_test[:2], **{kwarg_name: kwarg_value})


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_explain_methods_still_forward_genuinely_unknown_kwargs_to_plugin(method_name):
    """ADR-038 5C: a name unknown on every CalibratedExplainer surface is treated
    as plugin-defined and must NOT raise (preserves the §3 exception)."""
    cal_exp, x_test = _classification_explainer()
    method = getattr(cal_exp, method_name)
    # Should not raise; the unrecognized key is forwarded toward the plugin.
    result = method(x_test[:2], some_plugin_specific_key=123)
    assert isinstance(result, CalibratedExplanations)
    assert len(result) == 2


@pytest.mark.parametrize("method_name", ["explain_factual", "explore_alternatives"])
def test_explain_methods_log_forwarded_plugin_kwargs(method_name, caplog):
    cal_exp, x_test = _classification_explainer()
    method = getattr(cal_exp, method_name)

    with caplog.at_level(
        logging.INFO,
        logger="calibrated_explanations.core.calibrated_explainer",
    ):
        method(x_test[:2], some_plugin_specific_key=123)

    assert any(
        "forwarding explanation keyword arguments to plugins" in record.message
        for record in caplog.records
    )
    assert any("some_plugin_specific_key" in record.message for record in caplog.records)


def test_should_raise_configuration_error_when_removed_normalize_alias_passed_to_predict_proba():
    """ADR-038 5C: normalize= was previously not checked at all on
    CalibratedExplainer.predict_proba; it must now fail fast like every other
    removed alias."""
    cal_exp, x_test = _classification_explainer()
    with pytest.raises(ConfigurationError, match="removed in v0.11.5"):
        cal_exp.predict_proba(x_test[:2], normalize=True)


def test_should_raise_configuration_error_when_removed_guarded_kwarg_passed_to_predict():
    """ADR-038 5C: predict()/predict_proba() previously did not check removed
    guarded kwargs at all."""
    cal_exp, x_test = _classification_explainer()
    with pytest.raises(ConfigurationError, match="removed in v0.11.5"):
        cal_exp.predict(x_test[:2], guarded=True)


def test_should_raise_configuration_error_when_removed_alias_passed_to_explain_factual():
    """ADR-038 5C: explain_factual()/explore_alternatives() previously only
    checked removed guarded kwargs, not removed aliases or reject kwargs."""
    cal_exp, x_test = _classification_explainer()
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        cal_exp.explain_factual(x_test[:2], alpha=(1, 99))
    with pytest.raises(ConfigurationError, match="removed in v0.11.5"):
        cal_exp.explore_alternatives(x_test[:2], confidence=0.9)


def test_ce_skip_reject_still_works_on_predict_and_predict_proba():
    """ADR-038 5C: _ce_skip_reject is a real internal escape hatch used by
    core/explain/orchestrator.py and must remain allowed."""
    cal_exp, x_test = _classification_explainer()
    pred = cal_exp.predict(x_test[:2], _ce_skip_reject=True)
    proba = cal_exp.predict_proba(x_test[:2], _ce_skip_reject=True)
    assert len(pred) == 2
    assert len(proba) == 2


def test_plot_still_works_after_5c_predict_kwarg_scoping():
    """ADR-038 5C: plot()'s kwargs (style_override always re-injected, show=
    when passed) flow into predict()/predict_proba() via plotting.plot_global();
    predict() must strip them defensively like predict_proba() already does."""
    cal_exp, x_test = _classification_explainer()
    cal_exp.plot(x_test[:1], show=False)
    # plot() re-injects style_override and forwards show= into predict()/
    # predict_proba() through plot_global(); exercise that boundary directly to
    # prove predict() strips them instead of raising on the unknown names.
    pred = cal_exp.predict(x_test[:1], show=False, style_override=None)
    assert len(pred) == 1
