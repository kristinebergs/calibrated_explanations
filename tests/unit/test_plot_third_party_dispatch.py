"""Third-party plot-style dispatch contract (v1.0.0rc2).

Why a new test file?
--------------------
These tests cover the strict third-party dispatch contract introduced for
v1.0.0rc2: explicitly selected registered styles receive the complete,
unconsumed plugin request through ``FactualExplanation.plot``,
``AlternativeExplanation.plot``, and ``plot_global`` without monkey-patching
any CE callable. The behavior is a single coherent public contract spanning
three surfaces and does not belong to the PlotSpec default-promotion suite
(``test_plot_default_promotion.py``), which governs CE-owned style routing.

All plugins are registered through the public registry and ADR-006 trust
mechanisms; no CE plotting callable is replaced or wrapped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import calibrated_explanations.explanations.explanation as explanation_module
from calibrated_explanations import plotting
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.plugins.registry import (
    mark_plot_builder_trusted,
    mark_plot_renderer_trusted,
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
    reset_plugin_catalog,
)
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


def _plugin_meta(name: str, capability: str, *, trusted: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "version": "0.0.1",
        "provider": "ce-tests",
        "data_modalities": ("tabular",),
        "dependencies": (),
        "capabilities": [capability],
        "style": name,
        "output_formats": ["object"],
        "legacy_compatible": False,
        "supports_interactive": False,
        "trusted": trusted,
        "trust": trusted,
    }


class RecordingBuilder:
    def __init__(self, name: str, *, trusted: bool, error: Exception | None = None):
        self.plugin_meta = _plugin_meta(name, "plot:builder", trusted=trusted)
        self.contexts: list[Any] = []
        self.error = error

    def build(self, context: Any) -> Any:
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return {"synthetic_artifact": context.style}


class RecordingRenderer:
    def __init__(
        self,
        name: str,
        *,
        trusted: bool,
        result: Any = "renderer-result",
        error: Exception | None = None,
    ):
        self.plugin_meta = _plugin_meta(name, "plot:renderer", trusted=trusted)
        self.calls: list[tuple[Any, Any]] = []
        self.result = result
        self.error = error

    def render(self, artifact: Any, *, context: Any) -> Any:
        self.calls.append((artifact, context))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture()
def register_synthetic_style():
    """Register a synthetic style via the public registry; reset afterwards."""
    registered: list[str] = []

    def _register(
        style: str,
        *,
        trusted: bool = True,
        renderer_result: Any = "renderer-result",
        build_error: Exception | None = None,
        render_error: Exception | None = None,
        builder: Any = None,
    ) -> tuple[Any, RecordingRenderer]:
        builder = builder or RecordingBuilder(
            f"{style}.builder", trusted=trusted, error=build_error
        )
        renderer = RecordingRenderer(
            f"{style}.renderer", trusted=trusted, result=renderer_result, error=render_error
        )
        register_plot_builder(f"{style}.builder", builder, source="manual")
        register_plot_renderer(f"{style}.renderer", renderer, source="manual")
        if trusted:
            # ADR-006 keyed trust helpers: the public operator trust mechanism.
            mark_plot_builder_trusted(f"{style}.builder")
            mark_plot_renderer_trusted(f"{style}.renderer")
        register_plot_style(
            style,
            metadata={
                "style": style,
                "builder_id": f"{style}.builder",
                "renderer_id": f"{style}.renderer",
                "fallbacks": (),
                "legacy_compatible": False,
                "is_default": False,
                "default_for": (),
            },
        )
        registered.append(style)
        return builder, renderer

    yield _register
    reset_plugin_catalog(kind="plot")


@pytest.fixture()
def calibrated_wrapper() -> WrapCalibratedExplainer:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    wrapper = WrapCalibratedExplainer(LogisticRegression(random_state=0, solver="liblinear"))
    wrapper.fit(x[:20], y[:20])
    wrapper.calibrate(x[20:32], y[20:32])
    return wrapper


@pytest.fixture()
def calibrated_regression_wrapper() -> WrapCalibratedExplainer:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 3))
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.4, size=60)
    wrapper = WrapCalibratedExplainer(LinearRegression())
    wrapper.fit(x[:30], y[:30])
    wrapper.calibrate(x[30:50], y[30:50])
    return wrapper


@pytest.fixture()
def guard_builtin_instance_plotting(monkeypatch: pytest.MonkeyPatch):
    """Fail loudly if third-party dispatch leaks into built-in plotting."""

    def _fail(name: str):
        def _raiser(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(f"built-in {name} must not run for third-party styles")

        return _raiser

    monkeypatch.setattr(explanation_module, "plot_probabilistic", _fail("plot_probabilistic"))
    monkeypatch.setattr(explanation_module, "plot_regression", _fail("plot_regression"))
    monkeypatch.setattr(explanation_module, "plot_alternative", _fail("plot_alternative"))
    monkeypatch.setattr(explanation_module, "plot_triangular", _fail("plot_triangular"))
    monkeypatch.setattr(explanation_module, "calculate_metrics", _fail("calculate_metrics"))
    monkeypatch.setattr(plotting, "_require_matplotlib", _fail("_require_matplotlib"))


def _public_runtime_explainer(explanation: Any) -> Any:
    """Unwrap the explanation's frozen explainer snapshot via public accessors."""
    resolved = explanation.get_explainer()
    while not hasattr(resolved, "plugin_manager"):
        resolved = resolved.explainer
    return resolved


def _spy_rank_features(monkeypatch: pytest.MonkeyPatch, explanation: Any) -> list[int]:
    calls: list[int] = []
    original = explanation.rank_features

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(type(explanation), "rank_features", staticmethod(_spy))
    return calls


# ---------------------------------------------------------------------------
# Factual surface
# ---------------------------------------------------------------------------


def test_should_forward_complete_factual_request_when_third_party_style_selected(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
    monkeypatch,
) -> None:
    builder, renderer = register_synthetic_style("synthetic.factual")
    x_test = np.array([[0.1, -0.2, 0.3]])
    factual = calibrated_wrapper.explain_factual(x_test)[0]
    rank_calls = _spy_rank_features(monkeypatch, factual)

    result = factual.plot(
        filter_top=5,
        style="synthetic.factual",
        uncertainty=True,
        rnk_metric="ensured",
        rnk_weight=0.25,
        vendor_options={"nested": {"depth": 2}},
        show=False,
        path="out/custom.synthetic.bin",
        save_ext=["synthetic"],
    )

    assert result == "renderer-result"
    assert len(builder.contexts) == 1
    context = builder.contexts[0]
    assert context.explanation is factual
    assert context.intent["type"] == "factual"
    assert context.style == "synthetic.factual"
    assert context.options["filter_top"] == 5
    assert context.options["uncertainty"] is True
    assert context.options["rnk_metric"] == "ensured"
    assert context.options["rnk_weight"] == 0.25
    assert context.options["vendor_options"] == {"nested": {"depth": 2}}
    assert context.show is False
    assert context.path == "out/custom.synthetic.bin"
    assert context.save_ext == ("synthetic",)
    assert context.runtime["scope"] == "instance"
    # Instance explanations carry a frozen snapshot of the originating
    # explainer; the runtime reference resolves to that public snapshot.
    assert context.runtime["explainer"] is _public_runtime_explainer(factual)
    assert context.runtime["instance_index"] == factual.index
    assert renderer.calls[0][0] == {"synthetic_artifact": "synthetic.factual"}
    assert not rank_calls


def test_should_include_default_filter_top_none_when_factual_plugin_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual", show=False)

    assert builder.contexts[0].options["filter_top"] is None


def test_should_treat_none_renderer_result_as_handled_when_factual_plugin_returns_none(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, renderer = register_synthetic_style("synthetic.factual", renderer_result=None)
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(style="synthetic.factual", show=False)

    assert result is None
    assert len(builder.contexts) == 1
    assert len(renderer.calls) == 1


# ---------------------------------------------------------------------------
# Alternative surface
# ---------------------------------------------------------------------------


def test_should_forward_complete_alternative_request_when_third_party_style_selected(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
    monkeypatch,
) -> None:
    builder, renderer = register_synthetic_style("synthetic.alternative", renderer_result=None)
    alternative = calibrated_wrapper.explore_alternatives(np.array([[0.1, -0.2, 0.3]]))[0]
    rank_calls = _spy_rank_features(monkeypatch, alternative)

    result = alternative.plot(
        filter_top=8,
        style="synthetic.alternative",
        rnk_metric="ensured",
        rnk_weight=0.5,
        vendor_flag="on",
        show=False,
    )

    assert result is None
    context = builder.contexts[0]
    assert context.explanation is alternative
    assert context.intent["type"] == "alternative"
    assert context.style == "synthetic.alternative"
    assert context.options["filter_top"] == 8
    assert context.options["rnk_metric"] == "ensured"
    assert context.options["rnk_weight"] == 0.5
    assert context.options["vendor_flag"] == "on"
    assert context.runtime["scope"] == "instance"
    assert context.runtime["explainer"] is _public_runtime_explainer(alternative)
    assert len(renderer.calls) == 1
    assert not rank_calls


def test_should_not_rewrite_third_party_style_when_alternative_plugin_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("vendor.ensured.variant")
    alternative = calibrated_wrapper.explore_alternatives(np.array([[0.1, -0.2, 0.3]]))[0]

    alternative.plot(style="vendor.ensured.variant", show=False)

    assert builder.contexts[0].style == "vendor.ensured.variant"


# ---------------------------------------------------------------------------
# Global and dashboard surfaces
# ---------------------------------------------------------------------------


def test_should_forward_payload_options_and_runtime_when_global_style_selected(
    calibrated_wrapper,
    register_synthetic_style,
) -> None:
    builder, renderer = register_synthetic_style("synthetic.global")
    explainer = calibrated_wrapper.explainer
    x_test = np.array([[0.1, -0.2, 0.3], [0.5, 0.1, -0.4]])
    y_test = np.array([1, 0])

    result = explainer.plot(
        x_test,
        y_test,
        style="synthetic.global",
        aggregate_positions=True,
        global_options={"cards": ("a", "b")},
        show=False,
    )

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.intent["type"] == "global"
    assert context.options["aggregate_positions"] is True
    assert context.options["global_options"] == {"cards": ("a", "b")}
    payload = context.options["payload"]
    assert payload["proba"] is not None
    assert payload["x"] is x_test
    assert context.runtime["scope"] == "global"
    assert context.runtime["explainer"] is explainer
    assert context.runtime["x"] is x_test
    assert context.runtime["y"] is y_test
    assert context.runtime["threshold"] is None
    assert context.runtime["bins"] is None
    assert len(renderer.calls) == 1


def test_should_build_global_bounds_from_requested_regression_percentiles(
    calibrated_regression_wrapper,
    register_synthetic_style,
    monkeypatch,
) -> None:
    builder, _ = register_synthetic_style("synthetic.global.regression")
    explainer = calibrated_regression_wrapper.explainer
    x_test = np.array([[0.1, -0.2, 0.3], [0.5, 0.1, -0.4]])
    percentiles = (20, 80)
    expected_predict, (expected_low, expected_high) = explainer.predict(
        x_test, uq_interval=True, low_high_percentiles=percentiles
    )
    prediction_calls: list[dict[str, Any]] = []
    original_predict = explainer.predict

    def _record_predict(x: Any, **kwargs: Any) -> Any:
        prediction_calls.append(dict(kwargs))
        return original_predict(x, **kwargs)

    monkeypatch.setattr(explainer, "predict", _record_predict)

    explainer.plot(
        x_test,
        style="synthetic.global.regression",
        low_high_percentiles=percentiles,
        show=False,
    )

    payload = builder.contexts[0].options["payload"]
    assert prediction_calls == [
        {"uq_interval": True, "bins": None, "low_high_percentiles": percentiles}
    ]
    np.testing.assert_allclose(payload["predict"], expected_predict)
    np.testing.assert_allclose(payload["low"], expected_low)
    np.testing.assert_allclose(payload["high"], expected_high)
    assert np.all(np.asarray(payload["low"]) <= np.asarray(payload["predict"]))
    assert np.all(np.asarray(payload["predict"]) <= np.asarray(payload["high"]))


def test_should_reject_caller_supplied_payload_when_global_style_selected(
    calibrated_wrapper,
    register_synthetic_style,
) -> None:
    register_synthetic_style("synthetic.global")

    with pytest.raises(ValidationError, match="reserved"):
        calibrated_wrapper.explainer.plot(
            np.array([[0.1, -0.2, 0.3]]),
            style="synthetic.global",
            payload={"user": "value"},
            show=False,
        )


def test_should_let_trusted_dashboard_plugin_drive_public_explainer_when_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
) -> None:
    class DashboardBuilder:
        plugin_meta = _plugin_meta("synthetic.dashboard.builder", "plot:builder", trusted=True)

        def __init__(self) -> None:
            self.contexts: list[Any] = []

        def build(self, context: Any) -> Any:
            self.contexts.append(context)
            runtime = context.runtime
            explainer = runtime["explainer"]
            row = np.asarray(runtime["x"])[:1]
            factual = explainer.explain_factual(row)[0]
            alternatives = explainer.explore_alternatives(row)[0]
            return {
                "synthetic_dashboard": True,
                "factual_index": factual.index,
                "alternative_index": alternatives.index,
                "factual_options": dict(context.options.get("factual_options", {})),
            }

    builder = DashboardBuilder()
    _, renderer = register_synthetic_style("synthetic.dashboard", builder=builder)

    result = calibrated_wrapper.explainer.plot(
        np.array([[0.1, -0.2, 0.3], [0.5, 0.1, -0.4]]),
        style="synthetic.dashboard",
        factual_options={"filter_top": 4},
        alternative_options={"filter_top": 6},
        show=False,
    )

    assert result == "renderer-result"
    artifact = renderer.calls[0][0]
    assert artifact["synthetic_dashboard"] is True
    assert artifact["factual_index"] == 0
    assert artifact["alternative_index"] == 0
    assert artifact["factual_options"] == {"filter_top": 4}


# ---------------------------------------------------------------------------
# Strict resolution
# ---------------------------------------------------------------------------


def test_should_raise_actionable_error_when_explicit_style_is_unregistered(
    calibrated_wrapper,
    guard_builtin_instance_plotting,
) -> None:
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(ConfigurationError) as excinfo:
        factual.plot(style="vendor.missing", show=False)

    message = str(excinfo.value)
    assert "vendor.missing" in message
    assert "not registered" in message
    assert "global explanations" not in message


def test_should_raise_when_explicit_style_is_denied(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
    monkeypatch,
) -> None:
    register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]
    monkeypatch.setenv("CE_DENY_PLUGIN", "synthetic.factual")
    from calibrated_explanations.core.config_manager import (
        reset_process_config_manager_for_testing,
    )

    reset_process_config_manager_for_testing()
    try:
        with pytest.raises(ConfigurationError, match="denied"):
            factual.plot(style="synthetic.factual", show=False)
    finally:
        monkeypatch.delenv("CE_DENY_PLUGIN", raising=False)
        reset_process_config_manager_for_testing()


def test_should_raise_when_renderer_override_unregistered_for_explicit_style(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(ConfigurationError, match="renderer"):
        factual.plot(style="synthetic.factual", renderer="vendor.absent.renderer", show=False)


def test_should_withhold_runtime_context_when_explicit_style_is_untrusted(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.untrusted", trusted=False)
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(style="synthetic.untrusted", show=False)

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert dict(context.runtime) == {}


def test_should_surface_builder_error_without_fallback_when_explicit_style_fails(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.broken.build", build_error=RuntimeError("builder boom"))
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(RuntimeError, match="builder boom"):
        factual.plot(style="synthetic.broken.build", show=False)


def test_should_surface_renderer_error_without_fallback_when_explicit_style_fails(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.broken.render", render_error=RuntimeError("render boom"))
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(RuntimeError, match="render boom"):
        factual.plot(style="synthetic.broken.render", show=False)


def test_should_raise_when_explicit_style_conflicts_with_use_legacy(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(ValidationError, match="use_legacy"):
        factual.plot(style="synthetic.factual", use_legacy=True, show=False)


def test_should_raise_when_explicit_style_conflicts_with_string_style_override(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(ValidationError, match="style_override"):
        factual.plot(style="synthetic.factual", style_override="legacy", show=False)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_should_use_filename_as_context_path_without_rewriting_when_plugin_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual", filename="reports/plot.custom.ext")

    context = builder.contexts[0]
    assert context.path == "reports/plot.custom.ext"
    assert context.show is False
    assert "filename" not in context.options


def test_should_raise_when_path_and_filename_conflict_for_plugin_dispatch(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    with pytest.raises(ValidationError, match="path"):
        factual.plot(
            style="synthetic.factual", path="a/one.html", filename="b/two.html", show=False
        )


def test_should_accept_matching_path_and_filename_for_plugin_dispatch(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual", path="a/one.html", filename="a/one.html", show=False)

    assert builder.contexts[0].path == "a/one.html"


@pytest.mark.parametrize(
    ("transport", "expected_path", "expected_show"),
    [
        ({"filename": ""}, None, True),
        ({"path": ""}, None, True),
        ({"path": "out.html", "filename": ""}, "out.html", False),
        ({"path": "", "filename": "out.html"}, "out.html", False),
        ({"filename": "", "show": False}, None, False),
    ],
)
def test_should_treat_exact_empty_transport_paths_as_absent_when_plugin_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
    transport,
    expected_path,
    expected_show,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual", **transport)

    context = builder.contexts[0]
    assert context.path == expected_path
    assert context.show is expected_show


def test_should_default_show_true_when_no_output_path_for_plugin_dispatch(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual")

    context = builder.contexts[0]
    assert context.show is True
    assert context.path is None
    assert context.save_ext is None


def test_should_preserve_explicit_show_true_when_output_path_supplied(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(style="synthetic.factual", path="out/plot.html", show=True)

    context = builder.contexts[0]
    assert context.show is True
    assert context.path == "out/plot.html"


# ---------------------------------------------------------------------------
# Configured (non-explicit) selection channels: manager override, style_override
# ---------------------------------------------------------------------------
#
# A style selected without an explicit style="..." kwarg -- via
# PluginManager.plot_style_override, CE_PLOT_STYLE, pyproject.toml, or the
# plugin-dependency chain -- must dispatch through the same raw-request path
# as an explicit style, not the built-in option-consuming path. The
# explanation snapshot each explanation carries is frozen at explain_factual/
# explore_alternatives time, so the configured preference must be set on the
# live explainer BEFORE the explanation is generated.


def test_should_forward_complete_factual_request_when_style_selected_via_manager_override(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.factual")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = "synthetic.configured.factual"
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(
        filter_top=5, uncertainty=True, rnk_metric="ensured", rnk_weight=0.25, show=False
    )

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.style == "synthetic.configured.factual"
    assert context.options["filter_top"] == 5
    assert context.options["uncertainty"] is True
    assert context.options["rnk_metric"] == "ensured"
    assert context.options["rnk_weight"] == 0.25
    assert context.runtime["scope"] == "instance"


def test_should_forward_complete_alternative_request_when_style_selected_via_manager_override(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.alternative")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = (
        "synthetic.configured.alternative"
    )
    alternative = calibrated_wrapper.explore_alternatives(np.array([[0.1, -0.2, 0.3]]))[0]

    result = alternative.plot(filter_top=6, rnk_metric="ensured", show=False)

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.options["filter_top"] == 6
    assert context.options["rnk_metric"] == "ensured"


def test_should_forward_complete_global_request_when_style_selected_via_manager_override(
    calibrated_wrapper,
    register_synthetic_style,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.global")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = "synthetic.configured.global"

    result = calibrated_wrapper.explainer.plot(
        np.array([[0.1, -0.2, 0.3], [0.4, 0.1, -0.2]]),
        aggregate_positions=True,
        show=False,
    )

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.options["aggregate_positions"] is True
    assert "payload" in context.options
    assert context.runtime["scope"] == "global"


def test_should_treat_none_renderer_result_as_handled_when_configured_style_dispatched(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    register_synthetic_style("synthetic.configured.none", renderer_result=None)
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = "synthetic.configured.none"
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(show=False)

    assert result is None


def test_should_dispatch_configured_factual_style_when_use_legacy_false(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.factual.nonlegacy")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = (
        "synthetic.configured.factual.nonlegacy"
    )
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(use_legacy=False, filter_top=5, uncertainty=True, show=False)

    assert result == "renderer-result"
    assert builder.contexts[0].options["filter_top"] == 5
    assert builder.contexts[0].options["uncertainty"] is True


def test_should_dispatch_configured_alternative_style_when_use_legacy_false(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.alternative.nonlegacy")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = (
        "synthetic.configured.alternative.nonlegacy"
    )
    alternative = calibrated_wrapper.explore_alternatives(np.array([[0.1, -0.2, 0.3]]))[0]

    result = alternative.plot(use_legacy=False, filter_top=6, show=False)

    assert result == "renderer-result"
    assert builder.contexts[0].options["filter_top"] == 6


def test_should_treat_none_result_as_handled_for_configured_global_style_when_use_legacy_false(
    calibrated_wrapper,
    register_synthetic_style,
) -> None:
    builder, renderer = register_synthetic_style(
        "synthetic.configured.global.nonlegacy", renderer_result=None
    )
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = (
        "synthetic.configured.global.nonlegacy"
    )

    result = calibrated_wrapper.explainer.plot(
        np.array([[0.1, -0.2, 0.3], [0.4, 0.1, -0.2]]),
        use_legacy=False,
        show=False,
    )

    assert result is None
    assert len(builder.contexts) == 1
    assert len(renderer.calls) == 1


def test_should_bypass_configured_legacy_global_style_when_use_legacy_false(
    calibrated_wrapper,
    monkeypatch,
) -> None:
    explainer = calibrated_wrapper.explainer
    explainer.plugin_manager.plot_style_override = "legacy"
    resolved_styles: list[str | None] = []
    original_resolve = explainer.plugin_manager.resolve_plot_plugin

    def _record_resolve(*, explicit_style=None, renderer_override=None):
        resolved_styles.append(explicit_style)
        return original_resolve(
            explicit_style=explicit_style,
            renderer_override=renderer_override,
        )

    monkeypatch.setattr(explainer.plugin_manager, "resolve_plot_plugin", _record_resolve)

    explainer.plot(
        np.array([[0.1, -0.2, 0.3], [0.4, 0.1, -0.2]]),
        use_legacy=False,
        show=False,
    )

    assert resolved_styles == ["plot_spec.default"]


def test_should_not_dispatch_configured_style_when_use_legacy_true(
    calibrated_wrapper,
    register_synthetic_style,
    monkeypatch,
) -> None:
    builder, _ = register_synthetic_style("synthetic.configured.bypass")
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = "synthetic.configured.bypass"
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]
    monkeypatch.setattr(explanation_module, "plot_probabilistic", lambda *a, **k: "legacy-handled")

    result = factual.plot(use_legacy=True, show=False)

    assert result == "legacy-handled"
    assert not builder.contexts, "configured third-party style must not run when use_legacy=True"


def test_should_fall_through_silently_when_configured_style_is_unregistered(
    calibrated_wrapper,
    monkeypatch,
) -> None:
    calibrated_wrapper.explainer.plugin_manager.plot_style_override = "vendor.never.registered"
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]
    monkeypatch.setattr(
        explanation_module, "plot_probabilistic", lambda *a, **k: "fallback-handled"
    )

    result = factual.plot(show=False)

    assert result == "fallback-handled"


def test_should_forward_complete_factual_request_when_style_selected_via_style_override_kwarg(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.viaoverride.factual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    result = factual.plot(
        style_override="synthetic.viaoverride.factual",
        uncertainty=True,
        rnk_metric="ensured",
        show=False,
    )

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.style == "synthetic.viaoverride.factual"
    assert context.options["uncertainty"] is True


def test_should_forward_complete_alternative_request_when_style_selected_via_style_override_kwarg(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.viaoverride.alternative")
    alternative = calibrated_wrapper.explore_alternatives(np.array([[0.1, -0.2, 0.3]]))[0]

    result = alternative.plot(
        style_override="synthetic.viaoverride.alternative", filter_top=4, show=False
    )

    assert result == "renderer-result"
    context = builder.contexts[0]
    assert context.options["filter_top"] == 4


# ---------------------------------------------------------------------------
# Renderer-override trust recomputation (ADR-006)
# ---------------------------------------------------------------------------


def test_should_withhold_runtime_when_renderer_override_is_untrusted(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.overridetrust", trusted=True)
    untrusted_renderer = RecordingRenderer("synthetic.untrusted.renderer", trusted=False)
    register_plot_renderer("synthetic.untrusted.renderer.id", untrusted_renderer, source="manual")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(
        style="synthetic.overridetrust",
        renderer="synthetic.untrusted.renderer.id",
        show=False,
    )

    context = builder.contexts[0]
    assert dict(context.runtime) == {}


def test_should_keep_runtime_when_renderer_override_is_also_trusted(
    calibrated_wrapper,
    register_synthetic_style,
    guard_builtin_instance_plotting,
) -> None:
    builder, _ = register_synthetic_style("synthetic.overridetrust2", trusted=True)
    trusted_renderer = RecordingRenderer("synthetic.trusted.renderer", trusted=True)
    register_plot_renderer("synthetic.trusted.renderer.id", trusted_renderer, source="manual")
    mark_plot_renderer_trusted("synthetic.trusted.renderer.id")
    factual = calibrated_wrapper.explain_factual(np.array([[0.1, -0.2, 0.3]]))[0]

    factual.plot(
        style="synthetic.overridetrust2",
        renderer="synthetic.trusted.renderer.id",
        show=False,
    )

    context = builder.contexts[0]
    assert context.runtime["scope"] == "instance"
