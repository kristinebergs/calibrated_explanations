"""Centralized test-only accessors for CE/WCE internal state.

`pre-v4.md` finding S4-H1 (v0.11.6 Task 49) found that ADR-030 private-member
remediation had converted internal `CalibratedExplainer`/`WrapCalibratedExplainer`
state into public, mostly-mutable aliases whose only purpose was letting tests
touch private state (their own docstrings/comments said "for testing"). Those
aliases were removed outright rather than replaced with a new allowlisted
private-member accessor: the repository's private-member policy treats any
`_name` dot/`getattr` access in tests as permanent technical debt requiring a
`tests/legacy/` + `expiry` allowlist entry, and this test-only state has no
natural expiry, so this module instead reaches internal state through
reflection (`__dict__` / class-level function lookup) rather than attribute
syntax, so no allowlist entry is needed at all. Where a genuine public route
already exists (e.g. `explainer.plugin_manager.bridge_monitors`, or exercising
`enable_fast_mode()` instead of calling the private initializer directly),
call sites use that route directly instead of a wrapper here — only reach for
these helpers when there truly is no public equivalent.

Do not add new class-level public aliases to solve this in the future; extend
this module instead.
"""

from __future__ import annotations

from typing import Any, Dict


def _get_private_attr(obj: Any, name: str) -> Any:
    """Return instance attribute *name* via `__dict__`, bypassing attribute syntax."""
    return obj.__dict__[name]


def _set_private_attr(obj: Any, name: str, value: Any) -> None:
    """Set instance attribute *name* via `__dict__`, bypassing attribute syntax."""
    obj.__dict__[name] = value


def _delete_private_attr(obj: Any, name: str) -> None:
    """Delete instance attribute *name* via `__dict__` when present."""
    obj.__dict__.pop(name, None)


def _call_private_method(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call instance method *name* via a class-dict lookup, bypassing attribute syntax."""
    func = type(obj).__dict__[name]
    return func(obj, *args, **kwargs)


def initialize_fast_interval_learner(explainer: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke the explainer's fast-mode interval-learner initializer in isolation.

    Unlike ``explainer.enable_fast_mode()``, this does not also flip ``_fast``
    to True first — use it when a test needs to verify only the initializer
    plumbing without changing the explainer's fast-mode state.
    """
    return _call_private_method(
        explainer, "_initialize_interval_learner_for_fast_explainer", *args, **kwargs
    )


def get_lime_helper(explainer: Any) -> Any:
    """Return the explainer's LIME integration helper."""
    return _get_private_attr(explainer, "_lime_helper")


def set_lime_helper(explainer: Any, value: Any) -> None:
    """Replace the explainer's LIME integration helper."""
    _set_private_attr(explainer, "_lime_helper", value)


def delete_lime_helper(explainer: Any) -> None:
    """Remove the explainer's LIME integration helper attribute."""
    _delete_private_attr(explainer, "_lime_helper")


def get_shap_helper(explainer: Any) -> Any:
    """Return the explainer's SHAP integration helper."""
    return _get_private_attr(explainer, "_shap_helper")


def set_shap_helper(explainer: Any, value: Any) -> None:
    """Replace the explainer's SHAP integration helper."""
    _set_private_attr(explainer, "_shap_helper", value)


def delete_shap_helper(explainer: Any) -> None:
    """Remove the explainer's SHAP integration helper attribute."""
    _delete_private_attr(explainer, "_shap_helper")


def get_pyproject_explanations(explainer: Any) -> Dict[str, Any] | None:
    """Return the explainer's parsed pyproject explanation config."""
    return _get_private_attr(explainer.plugin_manager, "_pyproject_explanations")


def set_pyproject_explanations(explainer: Any, value: Dict[str, Any] | None) -> None:
    """Replace the explainer's parsed pyproject explanation config."""
    _set_private_attr(explainer.plugin_manager, "_pyproject_explanations", value)


def get_pyproject_intervals(explainer: Any) -> Dict[str, Any] | None:
    """Return the explainer's parsed pyproject interval config."""
    return _get_private_attr(explainer.plugin_manager, "_pyproject_intervals")


def set_pyproject_intervals(explainer: Any, value: Dict[str, Any] | None) -> None:
    """Replace the explainer's parsed pyproject interval config."""
    _set_private_attr(explainer.plugin_manager, "_pyproject_intervals", value)


def get_pyproject_plots(explainer: Any) -> Dict[str, Any] | None:
    """Return the explainer's parsed pyproject plot config."""
    return _get_private_attr(explainer.plugin_manager, "_pyproject_plots")


def set_pyproject_plots(explainer: Any, value: Dict[str, Any] | None) -> None:
    """Replace the explainer's parsed pyproject plot config."""
    _set_private_attr(explainer.plugin_manager, "_pyproject_plots", value)


def serialise_preprocessor_value(wrapper: Any, value: Any) -> Any:
    """Serialise a preprocessor value using the wrapper's internal serialiser."""
    return _call_private_method(wrapper, "_serialise_preprocessor_value", value)


def extract_preprocessor_snapshot(wrapper: Any, preprocessor: Any) -> dict[str, Any] | None:
    """Extract a snapshot of *preprocessor* using the wrapper's internal helper."""
    return _call_private_method(wrapper, "_extract_preprocessor_snapshot", preprocessor)


def build_preprocessor_metadata(wrapper: Any) -> dict[str, Any]:
    """Build preprocessor metadata using the wrapper's internal helper."""
    return _call_private_method(wrapper, "_build_preprocessor_metadata")


def pre_fit_preprocess(wrapper: Any, x: Any) -> Any:
    """Run the wrapper's internal pre-fit preprocessing step on *x*."""
    return _call_private_method(wrapper, "_pre_fit_preprocess", x)


def pre_transform(wrapper: Any, x: Any, stage: str = "predict") -> Any:
    """Run the wrapper's internal transform step on *x* for *stage*."""
    return _call_private_method(wrapper, "_pre_transform", x, stage=stage)


def maybe_preprocess_for_inference(wrapper: Any, x: Any) -> Any:
    """Run the wrapper's internal inference preprocessing step on *x*."""
    return _call_private_method(wrapper, "_maybe_preprocess_for_inference", x)


def finalize_fit(wrapper: Any, reinitialize: bool) -> Any:
    """Run the wrapper's internal fit-finalization step."""
    return _call_private_method(wrapper, "_finalize_fit", reinitialize)


def format_proba_output(wrapper: Any, proba: Any, uq_interval: bool) -> Any:
    """Format *proba* using the wrapper's internal formatter."""
    return _call_private_method(wrapper, "_format_proba_output", proba, uq_interval)


def normalize_auto_encode_flag(wrapper: Any) -> bool:
    """Return the wrapper's normalized auto-encode flag."""
    return _call_private_method(wrapper, "_normalize_auto_encode_flag")


def normalize_public_kwargs(wrapper: Any, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Normalize public kwargs using the wrapper's internal normalizer."""
    if payload is None:
        return _call_private_method(wrapper, "_normalize_public_kwargs", **kwargs)
    return _call_private_method(wrapper, "_normalize_public_kwargs", payload, **kwargs)
