"""Phase 1A calibration/interval-learner helper delegators.

This module contains thin wrapper functions that encapsulate calibration-related
logic from ``CalibratedExplainer`` without changing behavior. The explainer
instance is passed in and used directly to avoid re-wiring state.
"""

from __future__ import annotations

import os
import numpy as np

from types import MappingProxyType
from typing import Any, List, Mapping, Sequence, Tuple

from .._interval_regressor import IntervalRegressor
from .._VennAbers import VennAbers
from ..plugins.intervals import (
    ClassificationIntervalCalibrator,
    IntervalCalibratorContext,
    IntervalCalibratorPlugin,
    RegressionIntervalCalibrator,
)
from ..plugins.registry import (
    ensure_builtin_plugins,
    find_interval_descriptor,
    load_entrypoint_plugins,
)
from ..utils.perturbation import perturb_dataset
from .exceptions import ConfigurationError


def _split_csv(value: str | None) -> Tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _coerce_string_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,) if value else tuple()
    if isinstance(value, Sequence):
        collected: List[str] = []
        for item in value:
            if isinstance(item, str) and item:
                collected.append(item)
        return tuple(collected)
    return tuple()


def _collect_interval_hints(explainer: Any) -> Tuple[str, ...]:
    """Return ordered interval dependency hints gathered from explanation plugins."""

    hints: List[str] = []
    raw = getattr(explainer, "_interval_plugin_hints", {})
    if isinstance(raw, Mapping):
        for value in raw.values():
            for item in _coerce_string_tuple(value):
                if item and item not in hints:
                    hints.append(item)
    return tuple(hints)


def _build_interval_chain(
    explainer: Any, *, fast: bool, hints: Sequence[str]
) -> Tuple[str, ...]:
    """Assemble the ordered identifier chain for interval plugin resolution."""

    entries: List[str] = []
    override_attr = (
        "_fast_interval_plugin_override" if fast else "_interval_plugin_override"
    )
    override = getattr(explainer, override_attr, None)
    if isinstance(override, str) and override:
        entries.append(override)

    entries.extend(item for item in hints if item)

    env_prefix = "CE_INTERVAL_PLUGIN_FAST" if fast else "CE_INTERVAL_PLUGIN"
    entries.extend(_split_csv(os.environ.get(env_prefix)))

    env_fallbacks = os.environ.get(f"{env_prefix}_FALLBACKS")
    if env_fallbacks:
        entries.extend(_split_csv(env_fallbacks))
    elif not fast:
        entries.extend(_split_csv(os.environ.get("CE_INTERVAL_PLUGIN_FALLBACKS")))
    else:
        entries.extend(_split_csv(os.environ.get("CE_INTERVAL_PLUGIN_FAST_FALLBACKS")))

    py_settings = getattr(explainer, "_pyproject_plugins", {})
    if isinstance(py_settings, Mapping):
        key = "fast_interval" if fast else "interval"
        py_value = py_settings.get(key)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(_coerce_string_tuple(py_settings.get(f"{key}_fallbacks")))

    kw_attr = "_fast_interval_kw_fallbacks" if fast else "_interval_kw_fallbacks"
    entries.extend(_coerce_string_tuple(getattr(explainer, kw_attr, ())))

    default_identifier = "core.interval.fast" if fast else "core.interval.legacy"
    entries.append(default_identifier)

    seen: set[str] = set()
    ordered: List[str] = []
    for identifier in entries:
        if identifier and identifier not in seen:
            ordered.append(identifier)
            seen.add(identifier)
    return tuple(ordered)


def _resolve_interval_plugin(
    explainer: Any, *, fast: bool, hints: Sequence[str]
) -> Tuple[IntervalCalibratorPlugin, str | None]:
    """Return the interval plugin respecting overrides and trust settings."""

    override_attr = (
        "_fast_interval_plugin_override" if fast else "_interval_plugin_override"
    )
    override = getattr(explainer, override_attr, None)
    if override and not isinstance(override, str):
        plugin = override
        if not hasattr(plugin, "create"):
            raise ConfigurationError(
                "Interval plugin override must implement IntervalCalibratorPlugin"
            )
        return plugin, None

    chain = _build_interval_chain(explainer, fast=fast, hints=hints)
    errors: List[str] = []
    for identifier in chain:
        descriptor = find_interval_descriptor(identifier)
        if descriptor is None:
            errors.append(f"{identifier}: not registered")
            continue
        if not descriptor.trusted:
            errors.append(
                f"{identifier}: plugin is not trusted (use CE_TRUST_PLUGIN or mark_interval_trusted)"
            )
            continue
        plugin = descriptor.plugin
        if not hasattr(plugin, "create"):
            errors.append(
                f"{identifier}: registered object does not implement IntervalCalibratorPlugin"
            )
            continue
        return plugin, identifier

    tried = ", ".join(chain) if chain else "<none>"
    message = "Unable to resolve interval plugin"
    if fast:
        message += " for fast execution"
    message += f". Tried: {tried}"
    if errors:
        message += "; errors: " + "; ".join(errors)
    raise ConfigurationError(message)


def _build_legacy_interval_calibrator(explainer: Any) -> Any:
    """Construct the in-tree legacy calibrator for the current mode."""

    if explainer.mode == "classification":
        return VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
    if "regression" in explainer.mode:
        return IntervalRegressor(explainer)
    raise ConfigurationError(
        f"Unsupported interval mode '{explainer.mode}'. Expected classification or regression."
    )


def _build_fast_interval_calibrators(explainer: Any) -> List[Any]:
    """Construct the FAST calibrator list mirroring legacy behaviour."""

    calibrators: List[Any] = []
    X_cal, y_cal, bins = explainer.X_cal, explainer.y_cal, explainer.bins
    (
        explainer.fast_X_cal,
        explainer.scaled_X_cal,
        explainer.scaled_y_cal,
        scale_factor,
    ) = perturb_dataset(
        explainer.X_cal,
        explainer.y_cal,
        explainer.categorical_features,
        noise_type=explainer._CalibratedExplainer__noise_type,  # noqa: SLF001
        scale_factor=explainer._CalibratedExplainer__scale_factor,  # noqa: SLF001
        severity=explainer._CalibratedExplainer__severity,  # noqa: SLF001
        seed=getattr(explainer, "seed", None),
        rng=getattr(explainer, "rng", None),
    )
    explainer.bins = (
        np.tile(explainer.bins.copy(), scale_factor) if explainer.bins is not None else None
    )
    for feature_index in range(explainer.num_features):
        fast_X_cal = explainer.scaled_X_cal.copy()
        fast_X_cal[:, feature_index] = explainer.fast_X_cal[:, feature_index]
        if explainer.mode == "classification":
            calibrators.append(
                VennAbers(
                    fast_X_cal,
                    explainer.scaled_y_cal,
                    explainer.learner,
                    explainer.bins,
                    difficulty_estimator=explainer.difficulty_estimator,
                )
            )
        elif "regression" in explainer.mode:
            explainer.X_cal = fast_X_cal
            explainer.y_cal = explainer.scaled_y_cal
            calibrators.append(IntervalRegressor(explainer))

    explainer.X_cal, explainer.y_cal, explainer.bins = X_cal, y_cal, bins
    if explainer.mode == "classification":
        calibrators.append(
            VennAbers(
                explainer.X_cal,
                explainer.y_cal,
                explainer.learner,
                explainer.bins,
                difficulty_estimator=explainer.difficulty_estimator,
            )
        )
    elif "regression" in explainer.mode:
        calibrators.append(IntervalRegressor(explainer))
    return calibrators


def _build_interval_context(
    explainer: Any,
    *,
    base: Any,
    fast: bool,
    hints: Sequence[str],
) -> IntervalCalibratorContext:
    """Return the immutable context passed to interval plugins."""

    calibration_splits: Tuple[Tuple[Any, Any], ...] = (
        (explainer.X_cal, explainer.y_cal),
    )
    bins_mapping: Mapping[str, Any] = MappingProxyType({"bins": explainer.bins})
    residuals_mapping: Mapping[str, Any] = MappingProxyType(
        {"residuals": getattr(explainer, "residuals", None)}
    )
    difficulty_mapping: Mapping[str, Any] = MappingProxyType(
        {"estimator": explainer.difficulty_estimator}
    )
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "mode": explainer.mode,
            "predict_function": getattr(explainer, "predict_function", None),
            "calibrator": None if fast else base,
            "fast_calibrators": base if fast else None,
            "dependencies": tuple(hints),
            "pyproject": getattr(explainer, "_pyproject_plugins", {}),
        }
    )
    fast_flags: Mapping[str, Any] = MappingProxyType(
        {"fast": fast, "explainer_fast_mode": explainer.is_fast()}
    )
    return IntervalCalibratorContext(
        learner=explainer.learner,
        calibration_splits=calibration_splits,
        bins=bins_mapping,
        residuals=residuals_mapping,
        difficulty=difficulty_mapping,
        metadata=metadata,
        fast_flags=fast_flags,
    )


def _assert_interval_protocol(calibrator: Any, *, mode: str) -> Any:
    """Ensure returned calibrator satisfies the interval protocols."""

    candidate = calibrator
    if isinstance(candidate, Sequence) and candidate:
        candidate = candidate[-1]
    if not isinstance(candidate, ClassificationIntervalCalibrator):
        raise ConfigurationError(
            "Resolved interval plugin does not implement the classification interval protocol"
        )
    if "regression" in mode and not isinstance(candidate, RegressionIntervalCalibrator):
        raise ConfigurationError(
            "Resolved interval plugin does not implement the regression interval protocol"
        )
    return calibrator


def assign_threshold(explainer, threshold):
    """Thin wrapper around ``CalibratedExplainer.assign_threshold``.

    Exposed as a helper for tests and future extraction stages.
    """
    return explainer.assign_threshold(threshold)


def update_interval_learner(explainer, xs, ys, bins=None) -> None:
    """Mechanical move of ``CalibratedExplainer.__update_interval_learner``.

    Mirrors original semantics and exceptions exactly.
    """
    if explainer.is_fast():
        raise ConfigurationError("Fast explanations are not supported in this update path.")
    if explainer.mode == "classification":
        explainer.interval_learner = VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
    elif "regression" in explainer.mode:
        if isinstance(explainer.interval_learner, list):
            raise ConfigurationError("Fast explanations are not supported in this update path.")
        # update the IntervalRegressor
        explainer.interval_learner.insert_calibration(xs, ys, bins=bins)
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001


def initialize_interval_learner(explainer) -> None:
    """Initialise the primary interval calibrator via the plugin registry."""

    ensure_builtin_plugins()
    load_entrypoint_plugins()

    if explainer.is_fast():
        initialize_interval_learner_for_fast_explainer(explainer)
        return

    base_calibrator = _build_legacy_interval_calibrator(explainer)
    hints = _collect_interval_hints(explainer)
    plugin, identifier = _resolve_interval_plugin(explainer, fast=False, hints=hints)
    context = _build_interval_context(
        explainer,
        base=base_calibrator,
        fast=False,
        hints=hints,
    )
    calibrator = plugin.create(context, fast=False)
    explainer.interval_learner = _assert_interval_protocol(
        calibrator, mode=explainer.mode
    )
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

    resolved_identifier = identifier
    if resolved_identifier is None:
        plugin_meta = getattr(plugin, "plugin_meta", None)
        if isinstance(plugin_meta, Mapping):
            meta_name = plugin_meta.get("name")
            if meta_name:
                resolved_identifier = str(meta_name)
    if resolved_identifier:
        setattr(explainer, "_interval_plugin_identifier", resolved_identifier)


def initialize_interval_learner_for_fast_explainer(explainer) -> None:
    """Initialise the FAST interval calibrator via the plugin registry."""

    ensure_builtin_plugins()
    load_entrypoint_plugins()

    base_calibrators = _build_fast_interval_calibrators(explainer)
    hints = _collect_interval_hints(explainer)
    plugin, identifier = _resolve_interval_plugin(explainer, fast=True, hints=hints)
    context = _build_interval_context(
        explainer,
        base=base_calibrators,
        fast=True,
        hints=hints,
    )
    calibrator = plugin.create(context, fast=True)
    explainer.interval_learner = _assert_interval_protocol(
        calibrator, mode=explainer.mode
    )
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

    resolved_identifier = identifier
    if resolved_identifier is None:
        plugin_meta = getattr(plugin, "plugin_meta", None)
        if isinstance(plugin_meta, Mapping):
            meta_name = plugin_meta.get("name")
            if meta_name:
                resolved_identifier = str(meta_name)
    if resolved_identifier:
        setattr(explainer, "_fast_interval_plugin_identifier", resolved_identifier)


__all__ = ["assign_threshold", "initialize_interval_learner", "update_interval_learner"]
